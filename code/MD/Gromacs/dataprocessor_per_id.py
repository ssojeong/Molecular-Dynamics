import argparse
import torch


def data_processor_per_id(args):
    total_lines = args.pts * (args.num_mol * args.num_atom + 3)
    gro_tensor = torch.empty([args.pts, args.num_mol * args.num_atom, 3, 2])  # Assuming 6 values per atom line
    atom_id = [None] * args.num_mol * args.num_atom
    times = torch.empty([args.pts])
    with open(args.ip_path) as f:
        gro = f.readlines()
        assert len(gro) == total_lines, f"File length does not match expected ({total_lines})"

        for t in range(args.pts):
            frame_start = t * (args.num_mol * args.num_atom + 3)
            times[t] = float(gro[frame_start].strip().split()[3])
            for i in range(frame_start + 2, frame_start + 2 + args.num_mol * args.num_atom):
                line = gro[i].strip().split()

                if atom_id[i-frame_start-2] is None:
                    atom_id[i-frame_start-2] = (line[0], line[1])
                else:
                    assert line[0] == atom_id[i-frame_start-2][0], f'Molecule ID mismatch on line {i}: {line[0]} vs {mol_id[i, 0]}'
                    assert line[1] == atom_id[i-frame_start-2][1], 'Atom type does not match...'
                
                # -1.1 to ensure the q value range is [-1.1, 1.1]
                gro_tensor[t, i - (frame_start + 2), :, 0] = torch.tensor([float(x) for x in line[3:6]]) - 1.1
                gro_tensor[t, i - (frame_start + 2), :, 1] = torch.tensor([float(x) for x in line[6:]])

    # print(gro_tensor[:5, -1, :])
    # print(gro_tensor[-5:, 0, :])

    # Validation: No NaN in gro_tensor and times
    assert not torch.isnan(gro_tensor).any(), "gro_tensor contains NaN values!"
    assert not torch.isnan(times).any(), "times contains NaN values!"

    # Validation: No None in atom_id
    assert all(v is not None for v in atom_id), "atom_id list contains None values!"

    torch.save({"qp": gro_tensor,
                "times": times,
                "atom_id": atom_id}, args.op_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip_path', type=str, required=True, help='input .gro file path')
    parser.add_argument('--op_path', type=str, required=True, help='output .pt file path')
    parser.add_argument('--num_mol', type=int, default=8, help='number of molecules in each frame')
    parser.add_argument('--num_atom', type=int, default=3, help='atoms per molecule')
    parser.add_argument('--pts', type=int, default=451, help='number of time points')
    args = parser.parse_args()

    data_processor_per_id(args)

