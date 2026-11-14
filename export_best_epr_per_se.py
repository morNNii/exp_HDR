
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def read_pairwise_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def select_best_per_se(rows):
    groups = defaultdict(list)

    for row in rows:
        groups[int(row["se_index"])].append(row)

    results = []

    for se_index in sorted(groups):
        candidates = sorted(
            groups[se_index],
            key=lambda x: float(x["score"]),
            reverse=True
        )

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        best_score = float(best["score"])
        second_score = (
            float(second["score"]) if second else float("nan")
        )

        result = {
            "se_index": se_index,
            "se_file": best["se_file"],
            "se_gain_code": int(best["se_gain_code"]),
            "se_time_us": int(best["se_time_us"]),

            "selected_le_index": int(best["le_index"]),
            "selected_le_file": best["le_file"],
            "selected_le_gain_code": int(best["le_gain_code"]),
            "selected_le_time_us": int(best["le_time_us"]),

            "best_epr": float(best["epr"]),
            "best_score": best_score,

            "second_best_epr": (
                float(second["epr"]) if second else float("nan")
            ),
            "score_margin": (
                best_score - second_score if second
                else float("nan")
            ),

            "candidate_count": len(candidates),

            "mean_snr_db": float(best["mean_snr_db"]),
            "soft_information_coverage": float(
                best["soft_information_coverage"]
            ),
            "le_clip_mass": float(best["le_clip_mass"]),
            "both_clip_mass": float(best["both_clip_mass"]),
        }

        results.append(result)

    return results


def save_csv(path, results):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)


def save_txt(path, results):
    lines = []

    lines.append("BEST EPR / LE FOR EACH SE")
    lines.append("=" * 115)

    for r in results:
        lines.append(
            f"SE[{r['se_index']:03d}] "
            f"{r['se_file']}"
            f"  (Gain={r['se_gain_code']}, "
            f"Time={r['se_time_us']}us)"
        )

        lines.append(
            f"    -> LE[{r['selected_le_index']:03d}] "
            f"{r['selected_le_file']}"
            f"  (Gain={r['selected_le_gain_code']}, "
            f"Time={r['selected_le_time_us']}us)"
        )

        lines.append(
            f"       Best EPR={r['best_epr']:.4f}x  "
            f"Score={r['best_score']:.6f}  "
            f"SNR={r['mean_snr_db']:.3f} dB  "
            f"Candidates={r['candidate_count']}"
        )

        lines.append("-" * 115)

    path.write_text("\n".join(lines), encoding="utf-8")


def plot_results(results, output):
    se_index = np.array(
        [r["se_index"] for r in results]
    )

    best_epr = np.array(
        [r["best_epr"] for r in results]
    )

    best_score = np.array(
        [r["best_score"] for r in results]
    )

    le_gain = np.array(
        [r["selected_le_gain_code"] for r in results]
    )

    le_time = np.array(
        [r["selected_le_time_us"] for r in results]
    )

    margin = np.array(
        [r["score_margin"] for r in results]
    )

    fig, axes = plt.subplots(
        4, 1,
        figsize=(16, 13),
        sharex=True,
        constrained_layout=True
    )

    # 1. Selected EPR
    axes[0].plot(
        se_index, best_epr, "o-",
        markersize=3
    )
    axes[0].set_yscale("log", base=2)
    axes[0].set_ylabel("Best EPR")
    axes[0].set_title("Selected EPR for Each SE Frame")
    axes[0].grid(alpha=0.3)

    # 2. Selected LE Gain
    axes[1].step(
        se_index, le_gain,
        where="mid",
        label="LE Gain Code"
    )
    axes[1].set_ylabel("LE Gain Code")
    axes[1].set_title("Selected LE Gain")
    axes[1].grid(alpha=0.3)

    # 3. Selected LE Exposure Time
    axes[2].plot(
        se_index, le_time,
        "o-",
        markersize=3
    )
    axes[2].set_ylabel("LE Time (us)")
    axes[2].set_title("Selected LE Exposure Time")
    axes[2].grid(alpha=0.3)

    # 4. Best Score + Score Margin
    axes[3].plot(
        se_index, best_score,
        label="Best Score"
    )
    axes[3].set_ylabel("Best Score")
    axes[3].set_xlabel("SE Frame Index")
    axes[3].grid(alpha=0.3)

    ax_margin = axes[3].twinx()
    ax_margin.plot(
        se_index, margin,
        color="orange",
        alpha=0.6,
        label="Best - Second Best"
    )
    ax_margin.set_ylabel("Score Margin")

    axes[3].set_title("Best Score and Runner-up Margin")

    fig.suptitle(
        "Offline Pairwise AE: Best LE Selection per SE",
        fontsize=16
    )

    fig.savefig(
        output / "04_best_epr_per_se_detailed.png",
        dpi=160
    )

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input-csv",
        required=True,
        type=Path
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path
    )

    args = parser.parse_args()

    args.output_dir.mkdir(
        parents=True, exist_ok=True
    )

    rows = read_pairwise_csv(args.input_csv)

    if not rows:
        raise ValueError("pairwise_scores.csv is empty")

    results = select_best_per_se(rows)

    save_csv(
        args.output_dir / "best_epr_per_se_detailed.csv",
        results
    )

    save_txt(
        args.output_dir / "best_epr_per_se_detailed.txt",
        results
    )

    plot_results(results, args.output_dir)

    print(f"Processed {len(rows)} candidate pairs")
    print(f"SE frames with candidates: {len(results)}")
    print()

    print(
        f"{'SE':<28} "
        f"{'Selected LE':<28} "
        f"{'EPR':>10} "
        f"{'Score':>12}"
    )

    print("-" * 85)

    for r in results:
        print(
            f"{r['se_file']:<28} "
            f"{r['selected_le_file']:<28} "
            f"{r['best_epr']:>10.4f} "
            f"{r['best_score']:>12.6f}"
        )

    print()
    print("Finished:", args.output_dir)


if __name__ == "__main__":
    main()
