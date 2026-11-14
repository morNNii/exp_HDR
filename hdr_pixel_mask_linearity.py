#!/usr/bin/env python3
"""Paired Bayer RAW linearity validation for fixed-camera, static-scene captures.

Requires: numpy, matplotlib
Filename: GGGTTTTT_bayer.pgm (gain = GGG / gain_divisor, exposure in us).
PGM: binary P5, standard big-endian 16-bit samples when MAXVAL >= 256.

Outputs plots, paired-bin diagnostics CSV, summary JSON, optional masks.
No image registration: optional integer shift only. Align images beforehand if moved.
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


def read_pgm(path):
    comments = []
    with open(path, 'rb') as f:
        def token():
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError('Incomplete PGM header')
                if c == b'#':
                    comments.append(f.readline().decode('ascii', 'replace').strip())
                    continue
                if c in b' \r\n\t':
                    continue
                break
            buf = bytearray(c)
            while True:
                c = f.read(1)
                if not c:
                    raise ValueError('Incomplete PGM header token')
                if c in b' \r\n\t':
                    if c == b'\r':
                        p = f.tell()
                        if f.read(1) != b'\n':
                            f.seek(p)
                    return buf.decode('ascii')
                if c == b'#':
                    comments.append(f.readline().decode('ascii', 'replace').strip())
                    return buf.decode('ascii')
                buf.extend(c)
        magic = token()
        if magic != 'P5':
            raise ValueError(f'Expected P5 PGM: {path}')
        w, h, maxval = int(token()), int(token()), int(token())
        if not 1 <= maxval <= 65535:
            raise ValueError(f'Invalid PGM MAXVAL: {maxval}')
        dt = np.uint8 if maxval < 256 else np.dtype('>u2')
        size = w*h*np.dtype(dt).itemsize
        payload = f.read(size)
        if len(payload) != size:
            raise ValueError(f'Incomplete PGM pixels, expected {size} bytes, got {len(payload)}')
        if f.read(1):
            warnings.warn(f'{path.name}: trailing bytes after image')
    img = np.frombuffer(payload, dtype=dt).reshape(h, w).astype(np.uint16)
    info = {}
    for c in comments:
        if '=' in c:
            k,v = c.split('=',1)
            info[k.strip().upper()] = v.strip()
    return img, maxval, info


def load_frame(path, gain_source, gain_divisor):
    m = re.fullmatch(r'(\d{3})(\d{5})_bayer\.pgm',path.name,re.I)
    if m is None:
        raise ValueError(f'Filename should be GGGTTTTT_bayer.pgm: {path.name}')
    gain_file = int(m.group(1)) / gain_divisor
    exposure_us = int(m.group(2))
    image, maxval, header = read_pgm(path)
    gain_header = None
    exp_header = None
    try: gain_header = float(header['SENSOR_GAIN'])
    except (KeyError, ValueError): pass
    try: exp_header = float(header['EXP_TIME (US)'])
    except (KeyError, ValueError): pass
    if gain_header is not None and not np.isclose(gain_file,gain_header,rtol=.025,atol=.02):
        warnings.warn(f'{path.name}: filename gain {gain_file:g} != header SENSOR_GAIN {gain_header:g}; using {gain_source}')
    if exp_header is not None and not np.isclose(exp_header,exposure_us,rtol=.002,atol=1):
        warnings.warn(f'{path.name}: filename exposure {exposure_us}us != header {exp_header}us')
    gain = gain_file if gain_source == 'filename' else gain_header
    if gain is None or gain <= 0 or exposure_us <= 0:
        raise ValueError('Missing or nonpositive gain/exposure')
    return image, maxval, header, float(gain), exposure_us


def overlap_shift(short,long,dx,dy):
    """Long(x+dx, y+dy) corresponds to Short(x,y); crop both to same rectangle."""
    h,w = short.shape
    if long.shape != short.shape:
        raise ValueError(f'Short and long image sizes differ: {short.shape}, {long.shape}')
    x0,x1 = max(0,-dx), min(w,w-dx)
    y0,y1 = max(0,-dy), min(h,h-dy)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Shift exceeds image dimensions')
    return short[y0:y1,x0:x1],long[y0+dy:y1+dy,x0+dx:x1+dx],x0,y0


def channel_selection(shape, pattern, channel, x0, y0):
    h,w=shape
    pattern=pattern.upper()
    if pattern not in ('RGGB','BGGR','GBRG','GRBG'):
        raise ValueError('Invalid Bayer pattern')
    if channel == 'ALL':
        return np.ones(shape,bool)
    grid = np.array(list(pattern)).reshape(2,2)
    yy,xx=np.indices(shape)
    return grid[(yy+y0)%2,(xx+x0)%2] == channel


def float_or_none(x):
    return float(x) if np.isfinite(x) else None


def save_plot_pair(raw_s, raw_l, mask, out, a, bins, expected, limits):
    x=raw_s[mask].astype(np.float64)
    y=raw_l[mask].astype(np.float64)
    fig,ax=plt.subplots(figsize=(7.6,6.3))
    hb=ax.hexbin(x,y,gridsize=90,bins='log',mincnt=1,cmap='viridis',extent=(0,limits,0,limits))
    colorbar=fig.colorbar(hb,ax=ax)
    colorbar.set_label('log-scale pixel counts')
    xx=np.linspace(0,limits,200)
    ax.plot(xx,expected*xx,'--',lw=2,c='tomato',label=f'Theoretical y={expected:.5f}x')
    if a['ols_slope'] is not None:
        ax.plot(xx,a['ols_slope']*xx+a['ols_intercept_dn'],lw=1.7,c='cyan',label=f'OLS y={a["ols_slope"]:.5f}x+{a["ols_intercept_dn"]:.2f}')
    ax.set(xlim=(0,limits),ylim=(0,limits),xlabel='SHORT RAW DN (black-subtracted)',ylabel='LONG RAW DN (black-subtracted)',title=f'Paired RAW linearity | {len(x):,} common-valid pixels')
    ax.legend(fontsize=9,loc='upper left')
    ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out,dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--short-file',required=True)
    p.add_argument('--long-file',required=True)
    p.add_argument('--gain-source',choices=['filename','header'],default='filename')
    p.add_argument('--gain-divisor',type=float,default=32)
    p.add_argument('--black-level',type=float,default=0.0)
    p.add_argument('--white-level',type=float,default=None,help='Raw un-subtracted white code; default PGM MAXVAL')
    p.add_argument('--min-signal-dn',type=float,default=16)
    p.add_argument('--sat-margin-dn',type=float,default=2)
    p.add_argument('--bayer-pattern',default='BGGR',choices=['BGGR','RGGB','GRBG','GBRG'])
    p.add_argument('--channel',default='G',choices=['G','R','B','ALL'])
    p.add_argument('--roi',type=int,nargs=4,metavar=('X','Y','W','H'),help='ROI in SHORT frame pixel coordinates')
    p.add_argument('--shift-x',type=int,default=0,help='Long(x+shift-x,y+shift-y) aligns with short(x,y)')
    p.add_argument('--shift-y',type=int,default=0)
    p.add_argument('--scatter-max-samples',type=int,default=250000,help='Random sample count for plot, all common-valid pixels used for stats')
    p.add_argument('--dn-bin-width',type=int,default=16,help='Short RAW DN bin width for per-bin diagnostics')
    p.add_argument('--save-masks',action='store_true',help='Saves common-valid and individual-valid masks as compressed NPZ')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    s_path=args.input_dir / args.short_file
    l_path=args.input_dir / args.long_file
    raw_s,max_s,head_s,g_s,t_s=load_frame(s_path,args.gain_source,args.gain_divisor)
    raw_l,max_l,head_l,g_l,t_l=load_frame(l_path,args.gain_source,args.gain_divisor)
    raw_s,raw_l,ox,oy=overlap_shift(raw_s,raw_l,args.shift_x,args.shift_y)
    if args.roi:
        rx,ry,rw,rh=args.roi
        x0=max(0,rx-ox);y0=max(0,ry-oy)
        x1=min(raw_s.shape[1],rx+rw-ox);y1=min(raw_s.shape[0],ry+rh-oy)
        if x0>=x1 or y0>=y1: raise ValueError('ROI does not intersect valid shifted overlap')
        raw_s=raw_s[y0:y1,x0:x1]
        raw_l=raw_l[y0:y1,x0:x1]
        ox+=x0;oy+=y0
    reference=channel_selection(raw_s.shape,args.bayer_pattern,args.channel,ox,oy)
    white_s=args.white_level if args.white_level is not None else max_s
    white_l=args.white_level if args.white_level is not None else max_l
    dn_s=raw_s.astype(np.float32)-args.black_level
    dn_l=raw_l.astype(np.float32)-args.black_level
    # Do not clamp negative black-subtracted values before validity testing.
    v_s=(dn_s>=args.min_signal_dn)&(raw_s<white_s-args.sat_margin_dn)&reference
    v_l=(dn_l>=args.min_signal_dn)&(raw_l<white_l-args.sat_margin_dn)&reference
    common=v_s&v_l
    n_base=int(reference.sum());n_s=int(v_s.sum());n_l=int(v_l.sum());n=int(common.sum())
    if n<3:
        raise ValueError(f'Only {n} common-valid pixels. Lower threshold, choose overlapping exposure pair, or change ROI.')
    exp_s=t_s*g_s;exp_l=t_l*g_l
    expected=exp_l/exp_s
    x=dn_s[common].astype(np.float64)
    y=dn_l[common].astype(np.float64)
    irr_s=x/exp_s;irr_l=y/exp_l
    ratios=irr_l/irr_s
    residual=(irr_l-irr_s)/irr_s
    # OLS y = slope*x + intercept. When variance x is zero, regression undefined.
    var_x=np.var(x)
    slope=float(np.mean((x-x.mean())*(y-y.mean()))/var_x) if var_x>0 else np.nan
    intercept=float(y.mean()-slope*x.mean()) if np.isfinite(slope) else np.nan
    slope_zero=float(np.dot(x,y)/np.dot(x,x))
    corr=float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else np.nan
    med=float(np.median(ratios))
    q=np.quantile(ratios,[.05,.25,.5,.75,.95])
    abs_res=np.abs(residual)
    summary={
      'short_file':s_path.name,'long_file':l_path.name,
      'short_exposure_us':t_s,'long_exposure_us':t_l,'short_gain':g_s,'long_gain':g_l,
      'expected_raw_slope':expected,'expected_irradiance_ratio':1.0,
      'black_level':args.black_level,'white_level_short':white_s,'white_level_long':white_l,
      'min_signal_dn':args.min_signal_dn,'sat_margin_dn':args.sat_margin_dn,
      'channel':args.channel,'bayer_pattern':args.bayer_pattern,
      'shift_x':args.shift_x,'shift_y':args.shift_y,'roi':args.roi,
      'reference_pixels':n_base,'short_valid_pixels':n_s,'long_valid_pixels':n_l,'common_valid_pixels':n,
      'short_valid_fraction':n_s/n_base,'long_valid_fraction':n_l/n_base,'common_valid_fraction':n/n_base,
      'ols_slope':float_or_none(slope),'ols_intercept_dn':float_or_none(intercept),
      'zero_intercept_slope':float_or_none(slope_zero),'pearson_r':float_or_none(corr),
      'irradiance_ratio_median':med,'irradiance_ratio_p05':float(q[0]),'irradiance_ratio_p25':float(q[1]),
      'irradiance_ratio_p75':float(q[3]),'irradiance_ratio_p95':float(q[4]),
      'relative_residual_median':float(np.median(residual)),
      'relative_abs_residual_median':float(np.median(abs_res)),
      'relative_abs_residual_p95':float(np.quantile(abs_res,.95))
    }
    (args.output_dir/'linearity_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))
    # Plot downsampled points but use all for analysis.
    rng=np.random.default_rng(1234)
    if n>args.scatter_max_samples:
        choice=rng.choice(n,size=args.scatter_max_samples,replace=False)
        xx=x[choice];yy=y[choice];rr=ratios[choice];res=residual[choice]
    else:
        xx=x;yy=y;rr=ratios;res=residual
    lim=max(white_s,white_l)
    save_plot_pair(xx,yy,np.ones(len(xx),bool),args.output_dir/'01_raw_linearity.png',summary,100,expected,lim)
    fig,axs=plt.subplots(1,2,figsize=(13,4.8))
    # Relative ratio plot in logarithmic domain: robust to high outliers
    rlog=np.log2(np.maximum(rr,1e-12))
    cut=np.quantile(np.abs(rlog),.995)
    edge=max(1.,min(float(cut)*1.15,4.))
    axs[0].hist(rlog,bins=np.linspace(-edge,edge,150),density=True,color='#4C78A8')
    axs[0].axvline(0,color='tomato',linestyle='--',label='Ideal log2 ratio=0')
    axs[0].axvline(np.log2(med),color='black',label=f'Median ratio={med:.4f}')
    axs[0].set(xlabel='log2[(LONG / E_long) / (SHORT / E_short)]',ylabel='Density',title='Paired irradiance ratio (common-valid pixels)')
    axs[0].legend(fontsize=9)
    axs[1].hexbin(xx,res,gridsize=100,bins='log',mincnt=1,extent=(0,lim,-2,2),cmap='viridis')
    axs[1].axhline(0,color='tomato',linestyle='--')
    axs[1].set(xlabel='SHORT RAW DN',ylabel='(I_long-I_short)/I_short',title='Relative irradiance residual vs short RAW DN',ylim=(-1,2))
    for ax in axs: ax.grid(alpha=.15)
    fig.tight_layout();fig.savefig(args.output_dir/'02_irradiance_ratio_residual.png',dpi=180);plt.close(fig)
    # Per SHORT RAW DN bin diagnostics; no bias from unmatched pixel masks.
    width=args.dn_bin_width
    raw_bins=np.arange(0,max(white_s,white_l)+width,width,dtype=float)
    ids=np.searchsorted(raw_bins,x,side='right')-1
    rows=[]
    for bi in range(len(raw_bins)-1):
        ix=(ids==bi)
        if ix.sum()<30:continue
        rows.append({'short_dn_lo':float(raw_bins[bi]),'short_dn_hi':float(raw_bins[bi+1]),
                     'pixel_count':int(ix.sum()),'median_irradiance_ratio':float(np.median(ratios[ix])),
                     'median_relative_residual':float(np.median(residual[ix])),
                     'relative_abs_residual_p90':float(np.quantile(abs_res[ix],.90))})
    with open(args.output_dir/'per_short_dn_bins.csv','w',newline='',encoding='utf8') as f:
        wr=csv.DictWriter(f,fieldnames=['short_dn_lo','short_dn_hi','pixel_count','median_irradiance_ratio','median_relative_residual','relative_abs_residual_p90'])
        wr.writeheader();wr.writerows(rows)
    fig,ax=plt.subplots(figsize=(8.8,4.8))
    if rows:
        centers=np.array([(r['short_dn_lo']+r['short_dn_hi'])/2 for r in rows]);medians=np.array([r['median_irradiance_ratio'] for r in rows]);counts=np.array([r['pixel_count'] for r in rows]);
        ax.plot(centers,medians,'o-',ms=3,lw=1.5,label='Median (I_long/I_short)')
        ax2=ax.twinx();ax2.bar(centers,counts,width=width*.75,color='gray',alpha=.14,label='Pixels/bin');ax2.set_ylabel('Common-valid pixels / short DN bin')
    ax.axhline(1,color='tomato',ls='--',label='Ideal = 1')
    ax.set(xlabel='SHORT RAW DN bins',ylabel='Median irradiance ratio',title='Linearity vs RAW signal strength')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(args.output_dir/'03_ratio_by_short_dn.png',dpi=180);plt.close(fig)
    # Spatial map: 0 invalid, 1 short only, 2 long only, 3 both.
    state=np.zeros(raw_s.shape,np.uint8)
    state[v_s]=1;state[v_l]=2;state[common]=3
    from matplotlib.colors import ListedColormap,BoundaryNorm
    fig,ax=plt.subplots(figsize=(11,5.5))
    cmap=ListedColormap(['#222222','#5187c7','#e6a15a','#68b684'])
    ax.imshow(state,cmap=cmap,norm=BoundaryNorm(np.arange(-.5,4.5,1),4),interpolation='nearest')
    ax.set(title=f'Pixel validity (channel={args.channel}, common-valid={n/n_base:.1%})',xlabel='X within processed ROI',ylabel='Y within processed ROI')
    import matplotlib.patches as mpatches
    ax.legend(handles=[mpatches.Patch(color=cmap(i),label=label) for i,label in enumerate(['Invalid','Short only','Long only','Both valid'])],loc='upper right',fontsize=9)
    fig.tight_layout();fig.savefig(args.output_dir/'04_valid_pixel_mask.png',dpi=160);plt.close(fig)
    if args.save_masks:
        np.savez_compressed(args.output_dir/'valid_masks.npz',short=v_s,long=v_l,common=common)
    print('\nSaved results in:',args.output_dir.resolve())
    if args.shift_x==0 and args.shift_y==0:
        print('NOTE: No registration was performed; paired-pixel tests assume a stationary camera and scene.')


if __name__=='__main__':
    main()
