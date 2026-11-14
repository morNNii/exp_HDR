#!/usr/bin/env python3
"""2-exposure Bayer PGM irradiance and unregistered histogram fusion experiment.

Outputs:
  exposure_inventory.csv
  01_normalization_examples.png
  02_irradiance_histograms.png
  03_pair_diagnostics.png
  04_fused_scene_histogram.png
  05_fused_irradiance_bayer.png  (optional illustrative map)
  scene_histogram.csv, results.json, scene_histogram.npz

Install: pip install numpy matplotlib
Example: python hdr_irradiance_experiment.py --input-dir ./captures --output-dir ./results
"""
import argparse
import csv
import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@dataclass
class Frame:
    path: Path
    raw: np.ndarray
    exp_us: int
    gain: float
    gain_code: int
    header: dict
    white: int

    @property
    def exposure(self):
        return self.exp_us * self.gain


def read_pgm(path):
    """Read binary P5 PGM, with comments, 8-bit or >8-bit big-endian samples.

    The PGM standard uses big endian for 2-byte samples. Payload is not
    whitespace-stripped: first image byte is allowed to equal a whitespace byte.
    """
    comments = []
    with open(path, 'rb') as f:
        def next_token():
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError(f'Truncated PGM header: {path}')
                if c == b'#':
                    comments.append(f.readline().decode('ascii', errors='replace').strip())
                    continue
                if c in b' \t\r\n':
                    continue
                break
            token = bytearray(c)
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError(f'Truncated PGM header: {path}')
                if c == b'#':
                    comments.append(f.readline().decode('ascii', errors='replace').strip())
                    return token.decode('ascii')
                if c in b' \t\r\n':
                    # One whitespace separates header and payload. Handle CRLF.
                    if c == b'\r':
                        pos = f.tell()
                        if f.read(1) != b'\n':
                            f.seek(pos)
                    return token.decode('ascii')
                token.extend(c)

        magic = next_token()
        if magic != 'P5':
            raise ValueError(f'Expected binary P5 PGM, got {magic}: {path}')
        width, height, maxval = int(next_token()), int(next_token()), int(next_token())
        if not (0 < maxval <= 65535):
            raise ValueError(f'Unsupported PGM maxval {maxval}: {path}')
        dtype = np.uint8 if maxval < 256 else np.dtype('>u2')
        count = width * height
        size = count * np.dtype(dtype).itemsize
        payload = f.read(size)
        if len(payload) != size:
            raise ValueError(f'PGM payload incomplete: {path}, expected {size}, got {len(payload)}')
        if f.read(1):
            warnings.warn(f'{path.name}: trailing bytes beyond pixel payload')
        raw = np.frombuffer(payload, dtype=dtype).reshape(height, width).astype(np.uint16)
        header = {}
        for comment in comments:
            if '=' in comment:
                k, v = comment.split('=', 1)
                header[k.strip().upper()] = v.strip()
        return raw, maxval, header


def load_frame(path, args):
    match = re.match(r'^(\d{3})(\d{5})_bayer\.pgm$', path.name, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f'Unexpected filename (need GGGTTTTT_bayer.pgm): {path.name}')
    code, exp_us = int(match.group(1)), int(match.group(2))
    raw, maxval, header = read_pgm(path)
    file_gain = code / args.gain_divisor
    def header_float(key):
        try:
            return float(header[key])
        except (KeyError, ValueError):
            return None
    hgain = header_float('SENSOR_GAIN')
    hexp = header_float('EXP_TIME (US)')
    if hgain is not None and not np.isclose(hgain, file_gain, rtol=.025, atol=.02):
        warnings.warn(f'{path.name}: filename gain={file_gain:g}x, header SENSOR_GAIN={hgain:g}x; '
                      f'using {args.gain_source}. Verify filename/header meaning.')
    if hexp is not None and not np.isclose(hexp, exp_us, rtol=.002, atol=1):
        warnings.warn(f'{path.name}: filename exp={exp_us}us but header={hexp}us')
    if args.gain_source == 'header' and hgain is None:
        raise ValueError(f'{path.name} lacks usable SENSOR_GAIN header')
    gain = hgain if args.gain_source == 'header' else file_gain
    if gain <= 0 or exp_us <= 0:
        raise ValueError(f'Nonpositive exposure/gain: {path.name}')
    white = args.white_level if args.white_level is not None else maxval
    if white > maxval:
        raise ValueError(f'White level {white} > PGM maxval {maxval}')
    return Frame(path, raw, exp_us, gain, code, header, white)


def irradiance(frame, black_level):
    """Relative scene irradiance = black-subtracted DN/(exposure_us * relative_gain).

    Unit: DN/(us*gain-1). Not absolute physical electrons/second without alpha.
    """
    return np.maximum(frame.raw.astype(np.float32) - black_level, 0) / frame.exposure


def valid_mask(frame, args):
    """Pixels inside a reliable DN interval; upper bound excludes clipping."""
    signal = frame.raw.astype(np.float32) - args.black_level
    return (signal >= args.min_signal_dn) & (frame.raw < frame.white - args.sat_margin_dn)


def bayer_plane_masks(h, w, pattern):
    pattern = pattern.upper()
    if len(pattern) != 4 or sorted(pattern) != sorted('RGGB'):
        raise ValueError('Bayer pattern must be RGGB, BGGR, GRBG or GBRG')
    ret = {}
    for i, c in enumerate(pattern):
        mask = np.zeros((h, w), dtype=bool)
        mask[i // 2::2, i % 2::2] = True
        ret[f'{c}{i}'] = mask
    return ret


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a) & (a > 0)]
    if not len(a):
        return {'n': 0}
    q = np.percentile(a, [1, 5, 10, 50, 90, 95, 99])
    return {'n': int(a.size), **{f'p{p}': float(x) for p, x in zip([1,5,10,50,90,95,99], q)}}


def make_edges(frames, args):
    """Shared log-irradiance bins from reliable range of both frames."""
    ranges = []
    for fr in frames:
        lo = max(args.min_signal_dn, 0.5) / fr.exposure
        hi = (fr.white - args.sat_margin_dn - args.black_level) / fr.exposure
        if hi <= lo:
            raise ValueError('min_signal_dn / sat_margin_dn leaves no valid signal range')
        ranges.append((lo, hi))
    global_lo = min(x[0] for x in ranges)
    global_hi = max(x[1] for x in ranges)
    edges = np.geomspace(global_lo, global_hi, args.bins + 1)
    return edges, ranges


def histogram_per_frame(frame, edges, args, channel_mask=None):
    irr = irradiance(frame, args.black_level)
    v = valid_mask(frame, args)
    if channel_mask is not None:
        v &= channel_mask
        nbase = int(channel_mask.sum())
    else:
        nbase = irr.size
    counts = np.histogram(irr[v], bins=edges)[0].astype(np.float64)
    # Divide by ALL reference pixels, NOT only the valid ones.
    # Otherwise partial histograms have incompatible probability mass.
    return counts / nbase, int(v.sum()), nbase


def fused_histogram(frames, edges, ranges, args, channel_mask=None):
    """Fuse *distributions*, not corresponding pixels. No alignment required.

    Each source contributes where a scene irradiance would produce a reliable
    signal for that exposure. Histogram values are fractions of ALL pixels.
    Overlapping per-bin probabilities are averaged. Nonoverlap: one source.
    """
    centers = np.sqrt(edges[:-1] * edges[1:])
    probs, counts = [], []
    reliability = []
    for fr, (low, high) in zip(frames, ranges):
        p, nvalid, nbase = histogram_per_frame(fr, edges, args, channel_mask)
        probs.append(p)
        counts.append((nvalid, nbase))
        reliability.append((centers >= low) & (centers < high))
    probs, reliability = np.asarray(probs), np.asarray(reliability)
    contributors = np.sum(reliability, axis=0)
    fused = np.divide((probs * reliability).sum(axis=0), contributors,
                      out=np.zeros(args.bins, dtype=np.float64), where=contributors > 0)
    # Do not silently claim total=1: missing darkest/brightest radiance is possible.
    return fused, probs, contributors, counts


def log_irr_image(frame, black, clip_limits, max_display):
    arr = irradiance(frame, black)
    low, high = clip_limits
    arr = np.clip((np.log2(np.maximum(arr, low)) - np.log2(low)) / (np.log2(high) - np.log2(low)), 0, 1)
    # Use common physical/relative irradiance scale across all panels.
    return arr[::max_display, ::max_display]


def save_fig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches='tight')
    plt.close()


def plot_results(frames, edges, ranges, fused, probs, contributors, args, out):
    short, long = frames
    centers = np.sqrt(edges[:-1] * edges[1:])
    common_lo, common_hi = edges[0], edges[-1]
    # 1. Raw vs normalized images; same normalized scale (not individual auto-contrast).
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    for col, fr in enumerate(frames):
        thumbnail = fr.raw[::args.image_stride, ::args.image_stride]
        ax[0, col].imshow(thumbnail, cmap='gray', vmin=0, vmax=fr.white)
        ax[0, col].set_title(f'{"SHORT" if col==0 else "LONG"}: raw DN\n{fr.exp_us}us, gain={fr.gain:g}x')
        norm = log_irr_image(fr, args.black_level, (common_lo, common_hi), args.image_stride)
        norm[~valid_mask(fr, args)[::args.image_stride, ::args.image_stride]] = np.nan
        im = ax[1, col].imshow(norm, cmap='viridis', vmin=0, vmax=1)
        ax[1, col].set_title('Relative irradiance (log2, common range)\nInvalid clipped / dim pixels masked')
    for a in ax.ravel(): a.axis('off')
    fig.colorbar(im, ax=ax[1,:].tolist(), fraction=.027, pad=.02, label='Common log irradiance scale (0–1)')
    fig.savefig(out/'01_normalization_examples.png', dpi=160, bbox_inches='tight'); plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for idx, fr in enumerate(frames):
        v = valid_mask(fr, args)
        raw = fr.raw[v].astype(np.float64) - args.black_level
        hraw, eraw = np.histogram(raw, bins=100, range=(0, fr.white))
        ax[0].step(eraw[:-1], hraw / fr.raw.size, where='post', label=f'{fr.exp_us}us × {fr.gain:g}')
        ax[1].step(centers, probs[idx], where='mid', label=f'{fr.exp_us}us × {fr.gain:g}')
    ax[0].set(title='Original RAW distributions (valid DN only)', xlabel='RAW DN after black subtraction', ylabel='Fraction of ALL pixels / bin')
    ax[1].set(title='Exposure-normalized irradiance histograms (same bins)', xlabel='Relative irradiance [DN/(us × gain)]', ylabel='Fraction of ALL pixels / bin', xscale='log')
    for a in ax: a.grid(alpha=.25); a.legend()
    save_fig(out/'02_irradiance_histograms.png')

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    mask = (centers >= max(r[0] for r in ranges)) & (centers < min(r[1] for r in ranges))
    ax[0].step(centers, probs[0], label='SHORT')
    ax[0].step(centers, probs[1], label='LONG')
    ax[0].axvspan(max(r[0] for r in ranges), min(r[1] for r in ranges), alpha=.08, color='green') if np.any(mask) else None
    ax[0].set(xscale='log', xlabel='Relative irradiance', ylabel='Fraction of all pixels / bin', title='Overlap of normalized histograms')
    if np.any(mask):
        # Relative disagreement is a more useful diagnostic than pixel correlation for unregistered captures.
        a, b = probs[:, mask]
        tv = np.sum(np.abs(a-b)) / max(np.sum(a+b), 1e-12)
        ax[0].text(.04,.95, f'Overlap disagreement = {tv:.3f}\n(0 = identical)', transform=ax[0].transAxes, va='top')
    ax[1].step(centers, contributors, where='mid')
    ax[1].set(xscale='log', xlabel='Relative irradiance', ylabel='Number of usable exposures', ylim=(-.1, 2.2), yticks=[0,1,2], title='Reliability coverage by radiance bin')
    for a in ax: a.grid(alpha=.25)
    ax[0].legend(); save_fig(out/'03_pair_diagnostics.png')

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.step(centers, probs[0], where='mid', alpha=.65, label='Short RAW -> irradiance histogram')
    ax.step(centers, probs[1], where='mid', alpha=.65, label='Long RAW -> irradiance histogram')
    ax.step(centers, fused, where='mid', color='black', lw=1.8, label='FUSED scene histogram (no alignment)')
    ax.set(xscale='log', xlabel='Relative scene irradiance [DN/(us × gain)]', ylabel='Fraction of all pixels / log-bin', title='Two-RAW histogram fusion (per-bin reliability selection)')
    ax.grid(alpha=.25); ax.legend(); save_fig(out/'04_fused_scene_histogram.png')


def save_illustrative_bayer(frames, args, out):
    """OPTIONAL: illustrative pixel-wise irradiance composite, NOT used by histogram fusion.

    No spatial alignment: only meaningful if camera/scene are static; otherwise
    ghosting and incorrect point correspondences appear.
    """
    short, long = frames
    if short.raw.shape != long.raw.shape:
        warnings.warn('Dimensions differ; skip illustrative composite')
        return
    short_v = valid_mask(short, args)
    long_v = valid_mask(long, args)
    # Short for bright regions, long for shadows; prefer long where both valid.
    src_short = short_v & ~long_v
    src_long = long_v
    neither = ~(src_short | src_long)
    merged = np.full(short.raw.shape, np.nan, dtype=np.float32)
    lirr = irradiance(long, args.black_level)
    sirr = irradiance(short, args.black_level)
    merged[src_long] = lirr[src_long]
    merged[src_short] = sirr[src_short]
    # Scale shared with plotted individual normalized frames.
    positive = merged[np.isfinite(merged) & (merged > 0)]
    if positive.size == 0:
        return
    lo, hi = np.percentile(positive, [1,99])
    lo = max(lo, np.finfo(float).tiny)
    hi = max(hi, lo*1.01)
    fig, axs = plt.subplots(1, 3, figsize=(16,5))
    for i,(arr,valid,title) in enumerate([(sirr,short_v,'SHORT'),(lirr,long_v,'LONG'),(merged,~neither,'Illustrative pixel composite')]):
        thumb=arr[::args.image_stride,::args.image_stride].copy()
        thumb[~valid[::args.image_stride,::args.image_stride]]=np.nan
        im=axs[i].imshow(np.log2(np.clip(thumb,lo,hi)),cmap='viridis', vmin=np.log2(lo),vmax=np.log2(hi))
        axs[i].set_title(title);axs[i].axis('off')
    fig.colorbar(im,ax=axs.tolist(),fraction=.025,pad=.02,label='log2 relative irradiance')
    fig.suptitle('Pixel composite is illustrative ONLY: spatial alignment required for moving scenes')
    fig.savefig(out/'05_fused_irradiance_bayer.png',dpi=160,bbox_inches='tight');plt.close(fig)


def plot_all_exposures(all_frames, args, out):
    """Check irradiance scale consistency across the COMPLETE exposure sweep."""
    frames=sorted(all_frames,key=lambda f:f.exposure)
    rows=[]
    fig, ax=plt.subplots(1,2,figsize=(15,5.5))
    all_log=[]
    for fr in frames:
        vm=valid_mask(fr,args)
        vals=irradiance(fr,args.black_level)[vm]
        if vals.size:
            all_log.append(vals)
    if not all_log:
        return
    lo=max(min(np.min(v) for v in all_log if v.size),np.finfo(float).tiny)
    hi=max(np.max(v) for v in all_log if v.size)
    edges=np.geomspace(lo,hi,180)
    for fr in frames:
        v=valid_mask(fr,args)
        irr=irradiance(fr,args.black_level)
        q=stats(irr[v])
        rows.append({'filename':fr.path.name,'exp_us':fr.exp_us,'gain':fr.gain,
                     'exposure_us_gain':fr.exposure,'valid_fraction':float(v.mean()),
                     'saturation_fraction':float((fr.raw >= fr.white-args.sat_margin_dn).mean()),
                     'p10_irr':q.get('p10',float('nan')),
                     'p50_irr':q.get('p50',float('nan')),
                     'p90_irr':q.get('p90',float('nan'))})
        h=np.histogram(irr[v],bins=edges)[0]/irr.size
        ax[0].step(np.sqrt(edges[:-1]*edges[1:]),h,where='mid',alpha=.7,label=f'{fr.path.name[:8]}')
    x=np.arange(len(rows))
    ax[1].plot(x,[r['p50_irr'] for r in rows], 'o-',label='p50 (valid-only; can be biased)')
    ax[1].plot(x,[r['p10_irr'] for r in rows], 's--',label='p10 valid-only')
    ax[1].plot(x,[r['p90_irr'] for r in rows], '^--',label='p90 valid-only')
    ax[0].set(xscale='log',xlabel='Relative scene irradiance',ylabel='Fraction of ALL pixels / bin',title='All exposure images: normalized distributions')
    ax[1].set(yscale='log',xticks=x,xticklabels=[r['filename'][:8] for r in rows],xlabel='Filename prefix: gain code + exposure us',ylabel='Relative irradiance',title='Exposure sweep: percentile stability')
    ax[1].tick_params(axis='x',rotation=65)
    for a in ax:
        a.grid(alpha=.2)
        a.legend(fontsize=7,loc='best')
    fig.savefig(out/'06_all_exposures_validation.png',dpi=160,bbox_inches='tight');plt.close(fig)
    with open(out/'all_exposures_validation.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()));w.writeheader();w.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=Path('./hdr_results'))
    parser.add_argument('--gain-divisor',type=float,default=32.,help='Filename gain code / divisor; default 32')
    parser.add_argument('--gain-source',choices=['filename','header'],default='filename')
    parser.add_argument('--black-level',type=float,default=0.,help='Already corrected RAW: default 0 DN')
    parser.add_argument('--white-level',type=int,default=None,help='Raw saturation DN; default from PGM maxval')
    parser.add_argument('--sat-margin-dn',type=float,default=2.,help='Exclude DN >= white_level - margin (default 2)')
    parser.add_argument('--min-signal-dn',type=float,default=8.,help='Reliability floor above black in DN')
    parser.add_argument('--bins',type=int,default=256)
    parser.add_argument('--bayer-pattern',choices=['RGGB','BGGR','GRBG','GBRG'],default='BGGR')
    parser.add_argument('--channel',choices=['all','R','G','B'],default='all',help='Use one Bayer color for fusion statistics')
    parser.add_argument('--image-stride',type=int,default=4,help='Spatial stride for image previews')
    parser.add_argument('--short-file',type=str,default=None,help='Override automatically selected min exposure')
    parser.add_argument('--long-file',type=str,default=None,help='Override automatically selected max exposure')
    parser.add_argument('--save-pixel-composite',action='store_true',help='Optional UNALIGNED pixel composite illustration')
    args=parser.parse_args()
    if args.bins < 8 or args.gain_divisor<=0 or args.image_stride<1 or args.min_signal_dn<=0:
        parser.error('Invalid bins/gain-divisor/image-stride/min-signal-dn')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    paths=sorted(args.input_dir.glob('*_bayer.pgm'))
    if len(paths)<2:
        raise SystemExit('Need at least two *_bayer.pgm files')
    frames=[load_frame(p,args) for p in paths]
    with open(args.output_dir/'exposure_inventory.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['filename','exp_us','gain_code','gain_x','exposure_us_times_gain','white_DN','valid_fraction'])
        for fr in sorted(frames,key=lambda x:x.exposure):
            w.writerow([fr.path.name,fr.exp_us,fr.gain_code,fr.gain,fr.exposure,fr.white,float(valid_mask(fr,args).mean())])
    by_name={fr.path.name:fr for fr in frames}
    short=(by_name[args.short_file] if args.short_file else min(frames,key=lambda x:x.exposure))
    long=(by_name[args.long_file] if args.long_file else max(frames,key=lambda x:x.exposure))
    if short.path==long.path or short.exposure>=long.exposure:
        raise ValueError('Need different short and long with short exposure product < long')
    if short.raw.shape!=long.raw.shape:
        warnings.warn('Image shapes differ. Histogram fusion is still valid for identically sampled scene distributions, but less controlled.')
    # Use the SAME CFA color, or scene histogram may be biased by RGB spectral response.
    frames2=[short,long]
    masks=[]
    for fr in frames2:
        if args.channel=='all': masks.append(None)
        else:
            planes=bayer_plane_masks(*fr.raw.shape,args.bayer_pattern)
            cm=np.zeros(fr.raw.shape,dtype=bool)
            for name,mask in planes.items():
                if name[0]==args.channel: cm |= mask
            masks.append(cm)
    edges, ranges=make_edges(frames2,args)
    # The same channel mask is used on both frames; histogram fusion works without pixel correspondences.
    channel_mask=masks[0] if args.channel!='all' and short.raw.shape==long.raw.shape else None
    if args.channel!='all' and channel_mask is None:
        raise ValueError('Channel-specific comparison requires identical image sizes')
    fused, probs, contributors, counts=fused_histogram(frames2,edges,ranges,args,channel_mask)
    centers=np.sqrt(edges[:-1]*edges[1:])
    with open(args.output_dir/'scene_histogram.csv','w',newline='') as f:
        w=csv.writer(f);w.writerow(['irradiance_left','irradiance_center','irradiance_right','short_fraction','long_fraction','fused_fraction','contributors'])
        for i in range(args.bins):
            w.writerow([edges[i],centers[i],edges[i+1],probs[0,i],probs[1,i],fused[i],int(contributors[i])])
    np.savez_compressed(args.output_dir/'scene_histogram.npz', edges=edges, centers=centers, short_fraction=probs[0],long_fraction=probs[1],fused_fraction=fused,contributors=contributors)
    plot_results(frames2,edges,ranges,fused,probs,contributors,args,args.output_dir)
    plot_all_exposures(frames,args,args.output_dir)
    if args.save_pixel_composite:
        save_illustrative_bayer(frames2,args,args.output_dir)
    overlap=(centers >= max(x[0] for x in ranges)) & (centers < min(x[1] for x in ranges))
    overlap_disagreement=(float(np.abs(probs[0,overlap]-probs[1,overlap]).sum()/max((probs[0,overlap]+probs[1,overlap]).sum(),1e-12)) if overlap.any() else None)
    results={
        'definition':'relative_irradiance = max(raw_DN - black_level, 0)/(exposure_us * gain_x)',
        'unit':'DN / (microsecond * dimensionless_gain), not absolute physical photon irradiance',
        'short_file':short.path.name,'long_file':long.path.name,
        'short_exposure_product':short.exposure,'long_exposure_product':long.exposure,
        'exposure_ratio':long.exposure/short.exposure,
        'raw_image_shape_short':list(short.raw.shape),'raw_image_shape_long':list(long.raw.shape),
        'channel':args.channel,'bayer_pattern':args.bayer_pattern,
        'short_valid_fraction':counts[0][0]/counts[0][1],
        'long_valid_fraction':counts[1][0]/counts[1][1],
        'short_valid_irradiance_stats':stats(irradiance(short,args.black_level)[valid_mask(short,args)]),
        'long_valid_irradiance_stats':stats(irradiance(long,args.black_level)[valid_mask(long,args)]),
        'overlap_disagreement_0_ideal':overlap_disagreement,
        'hist_sum_short':float(probs[0].sum()),'hist_sum_long':float(probs[1].sum()),
        'hist_sum_fused':float(fused.sum()),
        'histogram_note':'Fused per-bin mass is not guaranteed to sum to 1; unobserved irradiance intervals are missing. Do not blindly renormalize.',
        'pixel_composite_note':'Not used for histogram fusion, and not reliable without image registration.'
    }
    (args.output_dir/'results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps({k:results[k] for k in ['short_file','long_file','exposure_ratio','short_valid_fraction','long_valid_fraction','overlap_disagreement_0_ideal','hist_sum_fused']},indent=2))
    print('Saved outputs to:', args.output_dir.resolve())

if __name__=='__main__':
    main()
