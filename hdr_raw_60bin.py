
import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from hdr_pixel_mask_linearity import read_pgm


def make_green_mask(shape, bayer="BGGR"):
    """
    Return mask of green pixels in Bayer RAW.
    """
    h, w = shape
    pattern = np.array(list(bayer)).reshape(2, 2)

    yy, xx = np.indices((h, w))
    return pattern[yy % 2, xx % 2] == "G"


def compute_histogram(raw, bins=60,
                      black_level=0,
                      channel="G",
                      bayer="BGGR"):

    # RAW is already black-level corrected.
    y = raw.astype(np.float64) - black_level

    # Select Bayer channel.
    if channel == "G":
        mask = make_green_mask(raw.shape, bayer)
        pixels = y[mask]
    elif channel == "ALL":
        pixels = y.reshape(-1)
    else:
        raise ValueError("channel must be G or ALL")

    # Simulated uniform 10-bit AE histogram.
    edges = np.linspace(0, 1024, bins + 1)

    # Clip only for histogram indexing.
    pixels = np.clip(pixels, 0, 1023)

    counts, _ = np.histogram(
        pixels,
        bins=edges
    )

    probability = counts / max(counts.sum(), 1)

    centers = (edges[:-1] + edges[1:]) / 2

    return edges, centers, counts, probability


def save_hist_csv(path, edges, counts, probability):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bin",
            "dn_left",
            "dn_right",
            "count",
            "probability"
        ])

        for i in range(len(counts)):
            writer.writerow([
                i,
                edges[i],
                edges[i + 1],
                int(counts[i]),
                probability[i]
            ])


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--short-file", required=True)
    parser.add_argument("--long-file", required=True)

    parser.add_argument("--bins", type=int, default=60)
    parser.add_argument("--black-level", type=float, default=0)
    parser.add_argument("--channel", default="G",
                        choices=["G", "ALL"])
    parser.add_argument("--bayer-pattern", default="BGGR",
                        choices=["RGGB", "BGGR", "GRBG", "GBRG"])

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    short, _, _ = read_pgm(input_dir / args.short_file)
    long, _, _ = read_pgm(input_dir / args.long_file)

    if short.shape != long.shape:
        raise ValueError("Short and Long image sizes differ")

    edges, centers, count_s, prob_s = compute_histogram(
        short, args.bins, args.black_level,
        args.channel, args.bayer_pattern
    )

    _, _, count_l, prob_l = compute_histogram(
        long, args.bins, args.black_level,
        args.channel, args.bayer_pattern
    )

    save_hist_csv(
        output_dir / "short_60bin.csv",
        edges, count_s, prob_s
    )

    save_hist_csv(
        output_dir / "long_60bin.csv",
        edges, count_l, prob_l
    )

    np.savez_compressed(
        output_dir / "ae_60bin_histograms.npz",
        edges=edges,
        centers=centers,
        short_counts=count_s,
        long_counts=count_l,
        short_probability=prob_s,
        long_probability=prob_l
    )

    fig, axes = plt.subplots(2, 1, figsize=(13, 9))

    x = np.arange(args.bins)

    axes[0].plot(
        x, prob_s, "o-",
        markersize=3,
        label="Short / Reference"
    )
    axes[0].plot(
        x, prob_l, "o-",
        markersize=3,
        label="Long / Comparison"
    )

    axes[0].set_title("Simulated AE Uniform 60-bin Histogram")
    axes[0].set_xlabel("AE Histogram Bin Index")
    axes[0].set_ylabel("Fraction of selected pixels")
    axes[0].set_xlim(0, args.bins - 1)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].stairs(
        prob_s, edges,
        label="Short / Reference"
    )
    axes[1].stairs(
        prob_l, edges,
        label="Long / Comparison"
    )

    axes[1].set_title("Same Histogram in RAW DN Domain")
    axes[1].set_xlabel("RAW DN")
    axes[1].set_ylabel("Fraction of selected pixels / bin")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "01_ae_60bin_comparison.png",
        dpi=180
    )
    plt.close()

    print("Short total pixels:", count_s.sum())
    print("Long total pixels :", count_l.sum())

    print("Short bin 0 percentage:",
          f"{prob_s[0] * 100:.2f}%")
    print("Long bin 0 percentage :",
          f"{prob_l[0] * 100:.2f}%")

    print("Finished:", output_dir)


if __name__ == "__main__":
    main()
