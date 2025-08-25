# plot_et_all_in_one.py
import glob
import matplotlib.pyplot as plt


def load_xvg_xy(path, cutoff=50.0):
    x, y = [], []
    with open(path) as f:
        for line in f:
            if not line or line[0] in ('@', '#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                t = float(parts[0])
                e = float(parts[1])
                if t <= cutoff:   # apply cutoff here
                    x.append(t)
                    y.append(e)
    return x, y


def main():
    files = sorted(glob.glob("ET_gap*.xvg"))
    if not files:
        raise FileNotFoundError("No ET_gap*.xvg files found.")

    plt.figure(figsize=(10, 6))
    for f in files:
        x, y = load_xvg_xy(f, cutoff=50.0)
        plt.plot(x, y, label=f"tau = {0.002 * int(f[6:-4]):.3f}")

    plt.xlabel("Time (ps)", fontsize=16)
    plt.ylabel("Energy", fontsize=16)
    plt.title("Energy vs Time (cutoff at t=50)", fontsize=20)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("energy_vs_time_all.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
