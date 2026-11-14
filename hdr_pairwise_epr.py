
#!/usr/bin/env python3
"""Offline pairwise EPR experiment: n RAWs -> n(n-1)/2 SE/LE candidates.

For a STATIC scene, reconstruct one common relative-irradiance histogram
from all 60-bin RAW histograms (offline oracle), then evaluate each pair
against the SAME scene histogram. No spatial registration is required.

This is signal-ratio (t*gain) testing, not a physical exposure-time ratio.
Alpha table and beta use the experimenter's assumed RAW-DN units.
"""
import argparse
import csv
import json
import re
import warnings
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ALPHA = {32: 105., 64: 147., 128: 195., 256: 256., 512: 361.}


def pgm_read(path):
    with path.open('rb') as f:
        def token():
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError(f'Incomplete PGM header: {path}')
                if c == b'#':
                    f.readline()
                    continue
                if c not in b' \t\r\n':
                    break
            data = bytearray(c)
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError('Incomplete PGM header token')
                if c in b' \t\r\n':
                    if c == b'\r':
                        pos = f.tell()
                        if f.read(1) != b'\n':
                            f.seek(pos)
                    return data.decode('ascii')
                data.extend(c)

        if token() != 'P5':
            raise ValueError(f'Not P5 PGM: {path}')

        width, height, maxval = (
            int(token()), int(token()), int(token())
        )

        if not 1 <= maxval <= 65535:
            raise ValueError('Invalid PGM maxval')

        dt = 'u1' if maxval < 256 else '>u2'
        size = width * height * (1 if maxval < 256 else 2)
        payload = f.read(size)

        if len(payload) != size:
            raise ValueError(f'Incomplete raster: {path}')

    return (
        np.frombuffer(payload, dtype=dt).reshape(height, width),
        maxval
    )


def parse_file(path, gain_divisor):
    m = re.fullmatch(
        r'(\d{3})(\d{5})_bayer\.pgm',
        path.name,
        re.I
    )

    if not m:
        raise ValueError(
            f'Expected GGGTTTTT_bayer.pgm: {path.name}'
        )

    gain_code, t = int(m[1]), int(m[2])

    if t <= 0 or gain_code <= 0:
        raise ValueError(path.name)

    return dict(
        name=path.name,
        path=path,
        gain_code=gain_code,
        time_us=t,
        gain=gain_code / gain_divisor,
        E=t * gain_code / gain_divisor
    )


def alpha(g):
    x = np.array(sorted(ALPHA), dtype=float)

    if g < x[0] or g > x[-1]:
        raise ValueError(
            f'Gain code {g} outside alpha calibration range'
        )

    return float(
        np.interp(g, x, [ALPHA[int(z)] for z in x])
    )


def raw_hist(path, bins, channel, pattern, black):
    raw, maxval = pgm_read(path)

    # Select CFA without constructing a full-size mask.
    if channel == 'G':
        p = np.array(list(pattern)).reshape(2, 2)

        pieces = [
            raw[y::2, x::2].ravel()
            for y in range(2)
            for x in range(2)
            if p[y, x] == 'G'
        ]

        pixels = np.concatenate(pieces).astype(np.float32)

    else:
        pixels = raw.astype(np.float32).ravel()

    pixels -= black

    if not np.isfinite(pixels).all():
        raise ValueError('Invalid pixel values')

    # Keep saturation in final histogram bin.
    pixels = np.clip(pixels, 0, 1023)

    counts, _ = np.histogram(
        pixels,
        bins=np.linspace(0, 1024, bins + 1)
    )

    return (
        counts.astype(float) / max(1, pixels.size),
        int(pixels.size),
        maxval
    )


def variance(y, gain_code, beta):
    return alpha(gain_code) * y + beta


def snr2(y, gain_code, beta):
    y = np.maximum(y, 0.)

    return y**2 / np.maximum(
        variance(y, gain_code, beta),
        1.e-12
    )


def reliability(y, gain_code, beta, white, soft):
    """Continuous low-SNR weighting, exact clipping at white-level.

    The low-SNR function is a modeling choice,
    NOT a hard DN threshold.
    """
    s2 = snr2(y, gain_code, beta)

    return (
        s2 / (s2 + soft)
    ) * (y < white)


def build_scene_hist(frames, irr_edges, args):
    """Common offline histogram; each frame's coarse DN bin is sub-sampled.

    Fusion weights are scene-bin measurement reliability, which is identical
    for all candidate comparisons. Unknown mass is not normalized away.
    """
    centers = np.sqrt(
        irr_edges[:-1] * irr_edges[1:]
    )

    accum = np.zeros(args.scene_bins)
    denom = np.zeros(args.scene_bins)

    dn_edges = np.linspace(
        0, 1024, args.ae_bins + 1
    )

    for fr in frames:

        # Uniform-within-DN-bin approximation:
        # critical near AE bin zero.
        h = np.zeros(args.scene_bins)

        for k, p in enumerate(fr['hist']):
            if p == 0:
                continue

            dn = (
                dn_edges[k]
                + (np.arange(args.subsamples) + .5)
                * (dn_edges[k + 1] - dn_edges[k])
                / args.subsamples
            )

            valid = (
                (dn > 0)
                & (dn < args.white_level)
            )

            if valid.any():
                h += np.histogram(
                    dn[valid] / fr['E'],
                    bins=irr_edges,
                    weights=np.full(
                        int(valid.sum()),
                        p / args.subsamples
                    )
                )[0]

        expected_y = centers * fr['E']

        w = reliability(
            expected_y,
            fr['gain_code'],
            args.beta,
            args.white_level,
            args.soft_scale
        )

        accum += h * w
        denom += w

    scene = np.divide(
        accum,
        denom,
        out=np.zeros_like(accum),
        where=denom > 1e-10
    )

    measured_mass = float(scene.sum())

    if measured_mass < 1e-12:
        raise ValueError(
            'Unable to estimate irradiance histogram'
        )

    # Approximate mass may exceed 1 due to
    # 60-bin rebinning/blur; NOT a coverage estimate.
    # Scores use this SAME histogram for all pairs.

    return centers, scene, measured_mass


def pair_metrics(se, le, irr, scene, args):
    ys = irr * se['E']
    yl = irr * le['E']

    s2s = snr2(
        ys, se['gain_code'], args.beta
    )

    s2l = snr2(
        yl, le['gain_code'], args.beta
    )

    ws = reliability(
        ys,
        se['gain_code'],
        args.beta,
        args.white_level,
        args.soft_scale
    )

    wl = reliability(
        yl,
        le['gain_code'],
        args.beta,
        args.white_level,
        args.soft_scale
    )

    merged = ws * s2s + wl * s2l

    utility = np.log1p(merged)

    # COMMON fixed scene histogram,
    # unconditional mass.
    # Not a probability re-normalization.
    score = float(scene @ utility)

    covered_weight = float(
        scene @ (1 - (1 - ws) * (1 - wl))
    )

    le_clip_mass = float(
        scene @ (yl >= args.white_level)
    )

    both_clip_mass = float(
        scene @ (
            (ys >= args.white_level)
            & (yl >= args.white_level)
        )
    )

    mean_snr_db = float(
        10 * np.log10(
            max(
                1e-12,
                (scene @ merged)
                / max(scene.sum(), 1e-12)
            )
        )
    )

    return dict(
        score=score,
        soft_information_coverage=covered_weight,
        le_clip_mass=le_clip_mass,
        both_clip_mass=both_clip_mass,
        mean_snr_db=mean_snr_db
    )


def save_rows(path, rows):
    with path.open(
        'w',
        newline='',
        encoding='utf-8-sig'
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0])
        )
        w.writeheader()
        w.writerows(rows)


def plot_results(frames, rows, centers, scene, output, args):

    # 1. Global pairwise scatter and per-SE optimum.
    x = np.array([r['epr'] for r in rows])
    y = np.array([r['score'] for r in rows])
    idx = np.array([r['se_index'] for r in rows])

    best = {}

    for r in rows:
        a = r['se_index']

        if a not in best or r['score'] > best[a]['score']:
            best[a] = r

    fig, ax = plt.subplots(
        2, 2,
        figsize=(15, 10),
        constrained_layout=True
    )

    sc = ax[0, 0].scatter(
        x, y,
        c=idx,
        cmap='viridis',
        s=16,
        alpha=.65
    )

    ax[0, 0].scatter(
        [v['epr'] for v in best.values()],
        [v['score'] for v in best.values()],
        s=46,
        facecolors='none',
        edgecolors='red',
        label='Best per SE'
    )

    ax[0, 0].set_xscale('log', base=2)

    ax[0, 0].set(
        xlabel='EPR = (tLE*gLE)/(tSE*gSE)',
        ylabel='Histogram-weighted log(1+SNR²)',
        title='All candidate SE -> later LE pairs'
    )

    ax[0, 0].legend()

    fig.colorbar(
        sc,
        ax=ax[0, 0],
        label='SE index'
    )

    for i in sorted(best):
        sub = [
            r for r in rows
            if r['se_index'] == i
        ]

        if len(sub) < 2:
            continue

        sub.sort(key=lambda r: r['epr'])

        ax[0, 1].plot(
            [r['epr'] for r in sub],
            [r['score'] for r in sub],
            'o-',
            ms=2.5,
            lw=1,
            alpha=.5
        )

    ax[0, 1].set_xscale('log', base=2)

    ax[0, 1].set(
        xlabel='EPR',
        ylabel='Score',
        title='Score curve for each SE (future frames as LE)'
    )

    ax[0, 1].grid(alpha=.2)

    # Plot common scene histogram.
    histogram_edges = np.r_[
        centers[0]**2 / centers[1],
        np.sqrt(centers[:-1] * centers[1:]),
        centers[-1]**2 / centers[-2]
    ]

    ax[1, 0].stairs(
        scene,
        histogram_edges,
        color='navy'
    )

    ax[1, 0].set_xscale('log')

    ax[1, 0].set(
        xlabel='Relative scene irradiance',
        ylabel='Fraction of all pixels/bin',
        title=(
            f'Common estimated histogram '
            f'(sum={scene.sum():.3f}; not coverage)'
        )
    )

    ax[1, 1].scatter(
        x,
        [r['soft_information_coverage'] for r in rows],
        s=10,
        alpha=.5,
        label='Soft coverage'
    )

    ax[1, 1].scatter(
        x,
        [r['le_clip_mass'] for r in rows],
        s=10,
        alpha=.5,
        label='LE clipped mass'
    )

    ax[1, 1].set_xscale('log', base=2)

    ax[1, 1].set(
        xlabel='EPR',
        ylabel='Fraction of all pixels (estimated)',
        title='Coverage / clipping tradeoff'
    )

    ax[1, 1].legend()
    ax[1, 1].grid(alpha=.2)

    fig.savefig(
        output / '01_epr_score_curves.png',
        dpi=160
    )

    plt.close(fig)

    # 2. Frame-index heatmap and best per-SE trajectory.
    n = len(frames)
    matrix = np.full((n, n), np.nan)

    for r in rows:
        matrix[
            r['se_index'],
            r['le_index']
        ] = r['score']

    fig, ax = plt.subplots(
        1, 2,
        figsize=(14, 5),
        constrained_layout=True
    )

    im = ax[0].imshow(
        np.ma.masked_invalid(matrix),
        origin='lower',
        aspect='auto',
        cmap='viridis'
    )

    ax[0].set(
        xlabel='Candidate LE frame index',
        ylabel='SE frame index',
        title='Score heatmap (upper triangle)'
    )

    fig.colorbar(
        im,
        ax=ax[0],
        label='Score'
    )

    ids = sorted(best)

    ax[1].plot(
        ids,
        [best[i]['epr'] for i in ids],
        'o-',
        label='Best EPR per SE'
    )

    ax[1].set_yscale('log', base=2)

    ax[1].set(
        xlabel='SE frame index',
        ylabel='Selected EPR',
        title='Offline EPR prediction by SE frame'
    )

    ax[1].grid(alpha=.3)
    ax[1].legend()

    fig.savefig(
        output / '02_epr_score_heatmap.png',
        dpi=160
    )

    plt.close(fig)

    # 3. For the first SE:
    # show candidate LE SNR vs irradiance.
    if best:
        i = sorted(best)[0]

        cand = [
            r for r in rows
            if r['se_index'] == i
        ]

        cand.sort(key=lambda r: r['epr'])

        choice = [
            cand[0],
            cand[len(cand) // 2],
            max(cand, key=lambda r: r['score']),
            cand[-1]
        ]

        fig, ax = plt.subplots(
            figsize=(10, 5),
            constrained_layout=True
        )

        unique_choices = {
            (z['se_index'], z['le_index']): z
            for z in choice
        }

        for r in unique_choices.values():
            se = frames[r['se_index']]
            le = frames[r['le_index']]

            ys = centers * se['E']
            yl = centers * le['E']

            s2 = (
                reliability(
                    ys,
                    se['gain_code'],
                    args.beta,
                    args.white_level,
                    args.soft_scale
                )
                * snr2(
                    ys,
                    se['gain_code'],
                    args.beta
                )
                +
                reliability(
                    yl,
                    le['gain_code'],
                    args.beta,
                    args.white_level,
                    args.soft_scale
                )
                * snr2(
                    yl,
                    le['gain_code'],
                    args.beta
                )
            )

            ax.plot(
                centers,
                10 * np.log10(np.maximum(s2, 1e-12)),
                label=f'EPR {r["epr"]:.2f}x'
            )

        ax.set_xscale('log')

        ax.set(
            xlabel='Relative irradiance',
            ylabel='Model merged SNR (dB)',
            title=(
                'SNR prediction vs scene brightness '
                f'(SE frame {i})'
            )
        )

        ax.legend()
        ax.grid(alpha=.25)

        fig.savefig(
            output / '03_snr_vs_irradiance.png',
            dpi=160
        )

        plt.close(fig)

    return best


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    p.add_argument(
        '--input-dir',
        required=True,
        type=Path
    )

    p.add_argument(
        '--output-dir',
        required=True,
        type=Path
    )

    p.add_argument(
        '--gain-divisor',
        type=float,
        default=32
    )

    p.add_argument(
        '--black-level',
        type=float,
        default=0
    )

    p.add_argument(
        '--white-level',
        type=float,
        default=1023
    )

    p.add_argument(
        '--bayer-pattern',
        choices=['RGGB', 'BGGR', 'GBRG', 'GRBG'],
        default='BGGR'
    )

    p.add_argument(
        '--channel',
        choices=['G', 'ALL'],
        default='G'
    )

    p.add_argument(
        '--ae-bins',
        type=int,
        default=60
    )

    p.add_argument(
        '--scene-bins',
        type=int,
        default=256
    )

    p.add_argument(
        '--subsamples',
        type=int,
        default=12
    )

    p.add_argument(
        '--beta',
        type=float,
        default=9
    )

    p.add_argument(
        '--soft-scale',
        type=float,
        default=1
    )

    p.add_argument(
        '--order',
        choices=['exposure', 'filename'],
        default='exposure'
    )

    p.add_argument(
        '--min-epr',
        type=float,
        default=1.0
    )

    p.add_argument(
        '--max-epr',
        type=float,
        default=512.0
    )

    p.add_argument(
        '--max-files',
        type=int,
        default=0
    )

    args = p.parse_args()

    if args.beta < 0 or args.soft_scale <= 0:
        raise ValueError(
            'beta >= 0, soft-scale > 0 required'
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    paths = list(
        args.input_dir.glob('*_bayer.pgm')
    )

    frames = [
        parse_file(q, args.gain_divisor)
        for q in paths
    ]

    if args.order == 'exposure':
        frames.sort(
            key=lambda x: x['E']
        )
    else:
        frames.sort(
            key=lambda x: x['name']
        )

    if args.max_files > 0:
        frames = frames[:args.max_files]

    if len(frames) < 2:
        raise ValueError(
            'At least 2 PGM frames required'
        )

    print(
        f'Loading {len(frames)} raw histograms...'
    )

    for f in frames:
        f['hist'], f['num_pixels'], maxval = raw_hist(
            f['path'],
            args.ae_bins,
            args.channel,
            args.bayer_pattern,
            args.black_level
        )

        if maxval != 1023:
            warnings.warn(
                f'{f["name"]}: PGM maxval={maxval}, '
                'expected 1023'
            )

        # Fail early if calibration is missing.
        alpha(f['gain_code'])

    # Scene domain includes signals from
    # the longest/shortest actual exposures.
    min_i = max(
        1e-9,
        0.25 / min(f['E'] for f in frames)
    )

    min_i = min(
        min_i,
        0.25 / max(f['E'] for f in frames)
    )

    max_i = 1024 / min(
        f['E'] for f in frames
    )

    edges = np.geomspace(
        min_i,
        max_i,
        args.scene_bins + 1
    )

    centers, scene, mass = build_scene_hist(
        frames,
        edges,
        args
    )

    if mass < 0.5:
        warnings.warn(
            f'Scene histogram sum {mass:.3f} is low; '
            'dark unobserved region may bias results'
        )

    rows = []

    # Pairwise SE -> LE evaluation.
    for i, se in enumerate(frames[:-1]):
        for j in range(i + 1, len(frames)):

            le = frames[j]
            epr = le['E'] / se['E']

            if not (
                args.min_epr <= epr <= args.max_epr
            ):
                continue

            met = pair_metrics(
                se,
                le,
                centers,
                scene,
                args
            )

            rows.append(
                dict(
                    se_index=i,
                    le_index=j,
                    se_file=se['name'],
                    le_file=le['name'],
                    se_time_us=se['time_us'],
                    se_gain_code=se['gain_code'],
                    le_time_us=le['time_us'],
                    le_gain_code=le['gain_code'],
                    epr=round(epr, 7),
                    **met
                )
            )

    if not rows:
        raise ValueError(
            'No valid EPR pairs; adjust '
            '--min-epr / --max-epr'
        )

    save_rows(
        args.output_dir / 'pairwise_scores.csv',
        rows
    )

    best = plot_results(
        frames,
        rows,
        centers,
        scene,
        args.output_dir,
        args
    )

    selected = [
        best[i]
        for i in sorted(best)
    ]

    save_rows(
        args.output_dir / 'best_per_se.csv',
        selected
    )

    np.savez_compressed(
        args.output_dir / 'scene_histogram.npz',
        edges=edges,
        centers=centers,
        scene_fraction=scene
    )

    meta = dict(
        frame_count=len(frames),
        pair_count=len(rows),
        estimated_histogram_sum_not_coverage=mass,
        max_scene_score_pair=max(
            rows,
            key=lambda r: r['score']
        ),
        assumptions=[
            'Static identical scene',
            'Uniform distribution inside each '
            'of 60 RAW DN bins',
            'Offline scene histogram uses all frames; '
            'NOT causal AE',
            'Gain alpha and beta use same signal '
            'units as 10-bit RAW',
            'EPR is time times gain signal ratio, '
            'NOT pure time ratio'
        ]
    )

    (
        args.output_dir / 'summary.json'
    ).write_text(
        json.dumps(meta, indent=2),
        encoding='utf8'
    )

    print(
        f'Frames: {len(frames)} | '
        f'Candidate pairs: {len(rows)} | '
        f'scene mass: {mass:.3f}'
    )

    top = meta['max_scene_score_pair']

    print(
        'Highest score pair:',
        top['se_file'],
        '->',
        top['le_file'],
        'EPR=',
        top['epr']
    )

    print('Output:', args.output_dir)


if __name__ == '__main__':
    main()
