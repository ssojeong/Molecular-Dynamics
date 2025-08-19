import torch
from typing import List, Tuple


class AtomPairIndexer:
    """
    Build index tensors for routing distances from a (N, N) atom–atom matrix
    into an (N, N, C) tensor with per-channel semantics for water (O, H1, H2).

    Channel mapping (C = 8):
      0 - Intra-molecular O→H   (same molecule)
      1 - Intra-molecular H→O
      2 - Intra-molecular H→H   (H1→H2 and H2→H1 within molecule)
      3 - Inter-molecular O→H   (different molecules)
      4 - Inter-molecular H→O
      5 - Inter-molecular H→H
      6 - Inter-molecular O→O
      7 - Self→Self (diagonal: i→i)

    Assumptions
    ----------
    - Each molecule has exactly 3 atoms ordered as (O, H1, H2).
    - There are `n_mol` molecules => total atoms N = 3 * n_mol.
    - Atoms are indexed globally: molecule m has atoms (3m, 3m+1, 3m+2).

    Typical use
    -----------
    >>> indexer = AtomPairIndexer(n_mol=8)
    >>> dist_mat = torch.rand(24, 24)
    >>> dist_tensor = indexer.fill_tensor(dist_mat)  # (24, 24, 8)
    """

    def __init__(self, n_mol: int) -> None:
        """
        Parameters
        ----------
        n_mol : int
            Number of water molecules (each is O,H,H).
        """
        self.n_mol: int = n_mol
        self.n_atoms: int = 3 * n_mol  # O, H1, H2 per molecule
        self.num_channels: int = 8

        # Per-channel index tensors (lengths differ per channel)
        self.rows: List[torch.LongTensor]
        self.cols: List[torch.LongTensor]

        # Flat (vectorized) indices across all channels for advanced indexing
        # shapes: (M,), (M,), (M,), where M = total number of indexed pairs
        self.rows_flat: torch.LongTensor
        self.cols_flat: torch.LongTensor
        self.ch_flat: torch.LongTensor

        self._build_indices()

    def _build_indices(self) -> None:
        """Construct per-channel and flattened (rows, cols, channel) index tensors."""
        # Atom indices per type (global indices)
        O_idx = torch.arange(0, self.n_atoms, 3, dtype=torch.long)
        H1_idx = torch.arange(1, self.n_atoms, 3, dtype=torch.long)
        H2_idx = torch.arange(2, self.n_atoms, 3, dtype=torch.long)

        rows: List[torch.LongTensor] = []
        cols: List[torch.LongTensor] = []

        # --- Intra channels (same molecule: i//3 == j//3) ---
        # 0: intra O→H (O→H1 and O→H2 within molecule)
        r = torch.cat([O_idx, O_idx], dim=0)
        c = torch.cat([H1_idx, H2_idx], dim=0)
        rows.append(r)
        cols.append(c)

        # 1: intra H→O (H1→O and H2→O within molecule)
        r = torch.cat([H1_idx, H2_idx], dim=0)
        c = torch.cat([O_idx, O_idx], dim=0)
        rows.append(r)
        cols.append(c)

        # 2: intra H→H (H1→H2 and H2→H1 within molecule)
        r = torch.cat([H1_idx, H2_idx], dim=0)
        c = torch.cat([H2_idx, H1_idx], dim=0)
        rows.append(r)
        cols.append(c)

        # Helper to build inter-molecular pairs via meshgrid and mask by molecule ID
        def inter_pairs(a: torch.LongTensor, b: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
            Ra, Cb = torch.meshgrid(a, b, indexing="ij")  # (len(a), len(b))
            # Different molecules: integer-divide by 3 to get molecule id
            mask = (Ra // 3) != (Cb // 3)
            return Ra[mask], Cb[mask]

        # --- Inter channels (different molecules: i//3 != j//3) ---
        # 3: inter O→H
        r, c = inter_pairs(O_idx, torch.cat([H1_idx, H2_idx]))
        rows.append(r)
        cols.append(c)

        # 4: inter H→O
        r, c = inter_pairs(torch.cat([H1_idx, H2_idx]), O_idx)
        rows.append(r)
        cols.append(c)

        # 5: inter H→H
        r, c = inter_pairs(torch.cat([H1_idx, H2_idx]), torch.cat([H1_idx, H2_idx]))
        rows.append(r)
        cols.append(c)

        # 6: inter O→O
        r, c = inter_pairs(O_idx, O_idx)
        rows.append(r)
        cols.append(c)

        # 7: self→self (diagonal)
        diag = torch.arange(self.n_atoms, dtype=torch.long)
        rows.append(diag)
        cols.append(diag)

        self.rows = rows
        self.cols = cols

        # Build flattened vectors for vectorized filling:
        ch_sizes = [len(r) for r in rows]
        self.rows_flat = torch.cat(rows, dim=0).long()
        self.cols_flat = torch.cat(cols, dim=0).long()
        self.ch_flat = torch.cat(
            [torch.full((sz,), ch) for ch, sz in enumerate(ch_sizes)],
            dim=0
        ).long()

    def get_indices(self, channel: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        """
        Return the (rows, cols) index tensors for a single channel.

        Parameters
        ----------
        channel : int
            Channel id in [0..7] per the mapping in the class docstring.

        Returns
        -------
        rows : torch.LongTensor
            Row indices for this channel (shape: (K,))
        cols : torch.LongTensor
            Column indices for this channel (shape: (K,))
        """
        return self.rows[channel], self.cols[channel]

    def fill_tensor(self, dist_mat: torch.Tensor) -> torch.Tensor:
        """
        Vectorized fill of a (B, N, N, 8, G) tensor from a (B, N, N, G) distance matrix.

        Parameters
        ----------
        dist_mat : torch.Tensor
            Shape (B, N, N, G), where N = 3 * n_mol.

        Returns
        -------
        out : torch.Tensor
            Shape (B, N, N, 8, G). Entries are written into the appropriate channels.
            Unused entries remain zero.
        """
        assert dist_mat.dim() == 4, f"dist_mat must be 4D (B,N,N,G), got {tuple(dist_mat.shape)}"
        B, N, N2, G = dist_mat.shape
        assert N == N2 == self.n_atoms, (
            f"dist_mat must be (B,{self.n_atoms},{self.n_atoms},G), got {tuple(dist_mat.shape)}"
        )

        # Indices on correct device
        rows = self.rows_flat.to(device=dist_mat.device)  # (M,)
        cols = self.cols_flat.to(device=dist_mat.device)  # (M,)
        ch = self.ch_flat.to(device=dist_mat.device)  # (M,)

        # Allocate output
        # out = torch.zeros((B, N, N, 8, G), dtype=dist_mat.dtype, device=dist_mat.device)
        out = -1 * torch.ones((B, N, N, 8, G), dtype=dist_mat.dtype, device=dist_mat.device)

        # Batch-wise advanced indexing
        # batch_idx: (B, M) pairs each batch with all (row,col,channel) indices
        batch_idx = torch.arange(B, device=dist_mat.device).view(-1, 1).expand(-1, rows.numel())

        # Write all groups/features at once with ":" on the last dim
        # print('!!!! grad before indexed r', out.requires_grad, dist_mat.requires_grad)
        out[batch_idx, rows, cols, ch, :] = dist_mat[:, rows, cols, :]
        # print('!!!! grad after indexed r', out.requires_grad, dist_mat.requires_grad)

        # ---- Sanity checks (numerical) ----
        # 1) Sum over channels (dim=3) should reconstruct dist_mat within tolerance for every group
        sum_ch = torch.relu(out).sum(dim=3)  # (B,N,N,G)
        torch.testing.assert_close(
            sum_ch, dist_mat, msg="Channel-sum (over 8) should equal dist_mat within tolerance"
        )
        return out


if __name__ == "__main__":
    import os

    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # build indexer (assumes your class has batch-aware fill_tensor: (B,N,N) -> (B,N,N,8))
    n_mol = 8  # → N = 24 atoms
    indexer = AtomPairIndexer(n_mol=n_mol)

    # Create a dummy distance matrix (B, N, N)
    B = 1
    N = indexer.n_atoms
    dist_mat = torch.rand(B, N, N, 6)

    # Make each (N,N) symmetric and zero-diagonal (per batch)
    dist_mat = 0.5 * (dist_mat + dist_mat.transpose(1, 2))

    # Fill into (B, N, N, 8)
    dist_tensor = indexer.fill_tensor(dist_mat)
    print("dist_tensor shape:", tuple(dist_tensor.shape))  # (B, N, N, 8)

    # Non-zero counts per channel (sum over batch, row, col)
    nz_per_ch = [(dist_tensor[..., ch, 0] != 0).sum().item() for ch in range(8)]
    print(f"Total entries: {B*N*N}; Non-zero entries per channel (0..7): {nz_per_ch}, sum: {sum(nz_per_ch)}")

    print("Intra-molecular O→H")
    print(dist_tensor[0, :9, :9, 0, -1])
    print("Intra-molecular H→O")
    print(dist_tensor[0, :9, :9, 1, -2])
    print("Intra-molecular H→H")
    print(dist_tensor[0, :9, :9, 2, -3])
    print("Inter-molecular O→H")
    print(dist_tensor[0, :9, :9, 3, -4])
    print("Inter-molecular H→O")
    print(dist_tensor[0, :9, :9, 4, -5])
    print("Inter-molecular H→H")
    print(dist_tensor[0, :9, :9, 5, -6])
    print("Inter-molecular O→O")
    print(dist_tensor[0, :9, :9, 6, 0])
    print("Atoms to themself")
    print(dist_tensor[0, :9, :9, 7, 2])

