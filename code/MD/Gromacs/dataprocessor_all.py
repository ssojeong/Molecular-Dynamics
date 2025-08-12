import argparse
import torch
import glob
import os

def combine_pt_files(args):
    
    pt_files = [os.path.join(args.ip_dir, f"{i:05d}.pt") for i in range(args.start, args.end)]

    tensors = []
    missing_tensors = []
    times = None
    atom_id = None
    traj_id_list = []
    for fpath in pt_files:
        traj_id = int(fpath[-8:-3])
        try:
            data = torch.load(fpath, map_location="cpu")
            tensors.append(data["qp"])
            traj_id_list.append(traj_id)
            if times is None:
                times = data["times"]
            else:
                assert torch.equal(times, data["times"]), "times does not match"
            
            if atom_id is None:
                atom_id = data["atom_id"]
            else:
                assert atom_id == data["atom_id"], "atom id does not match"
        
        except FileNotFoundError:
            missing_tensors.append(traj_id)

    # Combine along a new first dimension
    combined = torch.stack(tensors, dim=0)
    print(f"Combined tensor shape:", combined.shape)
    
    # Change the position range from [0, 2.2] to [-1.1, 1.1]
    # combined[:, :, :, :3] -= 1.1

    # Save to output file
    torch.save({"qp": combined,
                "times": times,
                "traj_id": torch.tensor(traj_id_list),
                "atom_id": atom_id}, args.op_path)
    print(f"Saved combined tensor to {args.op_path}")

    # Save missing indices to a log file
    if missing_tensors:
        log_path = args.op_path + ".missing.txt"
        with open(log_path, "w") as f:
            f.write("\n".join(map(str, missing_tensors)))
        print(f"Missing file indices saved to {log_path}")
    else:
        print("No missing files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip_dir', type=str, required=True, help='input directory containing .pt files')
    parser.add_argument('--op_path', type=str, required=True, help='output .pt file path')
    parser.add_argument('--start', type=int, default=1, help='start index of .pt files to combine')
    parser.add_argument('--end', type=int, default=50000, help='end index (exclusive) of .pt files to combine')
    args = parser.parse_args()

    combine_pt_files(args)

