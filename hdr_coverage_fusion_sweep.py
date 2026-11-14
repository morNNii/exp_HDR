
#!/usr/bin/env python3
"""2-exposure HDR: fixed-short / long-exposure sweep with validity-aware hist fusion.

Dependencies: numpy matplotlib
Example:
 python hdr_coverage_fusion_sweep.py --input-dir D:/HDR_Test/Pattern_01 \
   --output-dir D:/HDR_Test/FusionSweep --short-file 03208111_bayer.pgm \
   --long-files 03216049_bayer.pgm 03233000_bayer.pgm \
   --channel G --min-signal-dn 16 --sat-margin-dn 2

In scene histogram, each RAW exposure generates an UNCONDITIONAL pixel-fraction histogram,
not one normalized by its valid-pixel count. Each irradiance bin is selected from Long,
Short, or the average where both have valid measurement range (as in the paper).
Missing dark-range mass is NOT silently normalized away. Pixel coverage maps/percentages
assume images are spatially registered; distribution-only fusion does not require registration.
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
from matplotlib.colors import ListedColormap, BoundaryNorm


def pgm_read(path):
    metadata = {}
    with open(path, 'rb') as f:
        def tok():
            while True:
                ch = f.read(1)
                if not ch: raise ValueError('Truncated PGM header')
                if ch == b'#':
                    line = f.readline().decode('ascii','replace')
                    if '=' in line:
                        k,v=line.split('=',1);metadata[k.strip().upper()]=v.strip()
                    continue
                if ch in b' \t\n\r':continue
                break
            out=bytearray(ch)
            while True:
                ch=f.read(1)
                if not ch:raise ValueError('Unexpected EOF in token')
                if ch in b' \n\r\t':
                    if ch == b'\r':
                        p=f.tell()
                        if f.read(1)!=b'\n':f.seek(p)
                    break
                out.extend(ch)
            return out.decode('ascii')
        if tok()!='P5':raise ValueError(f'Not binary P5 PGM: {path}')
        width,height,maxval=int(tok()),int(tok()),int(tok())
        if not (1<=maxval<=65535): raise ValueError('Bad maxval')
        dtype=np.uint8 if maxval<256 else np.dtype('>u2')
        size=width*height*np.dtype(dtype).itemsize
        raw=f.read(size)
        if len(raw)!=size:raise ValueError(f'Incomplete PGM: expected {size} bytes, received {len(raw)}')
        if f.read(1):warnings.warn(f'{path.name}: trailing data after PGM payload')
    return np.frombuffer(raw,dtype=dtype).reshape(height,width).astype(np.uint16),maxval,metadata


def parse_frame(path,args):
    match=re.fullmatch(r'(\d{3})(\d{5})_bayer\.pgm',path.name,re.I)
    if not match:raise ValueError(f'Bad filename convention: {path.name}')
    g_file=int(match.group(1))/args.gain_divisor
    t_file=int(match.group(2))
    img,maxval,meta=pgm_read(path)
    gh=None;th=None
    try:gh=float(meta['SENSOR_GAIN'])
    except (KeyError,ValueError):pass
    try:th=float(meta['EXP_TIME (US)'])
    except (KeyError,ValueError):pass
    if gh is not None and not np.isclose(g_file,gh,rtol=.025,atol=.02):
        warnings.warn(f'{path.name}: filename gain {g_file:g} differs from header {gh:g}')
    if th is not None and not np.isclose(t_file,th,rtol=.002,atol=1):
        warnings.warn(f'{path.name}: filename time {t_file}us differs from header {th}us')
    g=gh if args.gain_source=='header' else g_file
    if g is None or g<=0 or t_file<=0:raise ValueError(f'Invalid exposure/gain: {path}')
    return dict(name=path.name, image=img, maxval=maxval, t=t_file, gain=g, E=t_file*g)


def channel_mask(shape,pattern,channel,roi_x,roi_y):
    if channel=='ALL':return np.ones(shape,bool)
    cfa=np.array(list(pattern)).reshape(2,2)
    yy,xx=np.indices(shape)
    return cfa[(yy+roi_y)%2,(xx+roi_x)%2]==channel


def data_roi(image,roi):
    if roi is None:return image,0,0
    x,y,w,h=roi
    if x<0 or y<0 or w<=0 or h<=0 or x+w>image.shape[1] or y+h>image.shape[0]:
        raise ValueError(f'ROI outside image: {roi} / {image.shape}')
    return image[y:y+h,x:x+w],x,y


def thresholds(frame,args):
    white=args.white_level if args.white_level is not None else frame['maxval']
    high=white-args.sat_margin_dn-args.black_level
    if high<=args.min_signal_dn:raise ValueError('DN upper bound below minimum DN')
    return float(high)


def counts_of_masks(ref,s,l):
    n=int(ref.sum())
    if n==0:raise ValueError('No selected channel samples')
    c=[int((ref & ~s & ~l).sum()),int((ref & s & ~l).sum()),
       int((ref & ~s & l).sum()),int((ref & s & l).sum())]
    return n,c,np.array(c,dtype=float)/n


def hist_irr(image,ref,valid,E,edges,nbase):
    v=(image[valid].astype(np.float64))/E
    hist=np.histogram(v,bins=edges)[0].astype(float)/nbase
    return hist


def save_csv(path,fieldnames,rows):
    with open(path,'w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=fieldnames);w.writeheader();w.writerows(rows)


def pair_analysis(short,long,ref,args,edges,out_dir):
    out_dir.mkdir(parents=True,exist_ok=True)
    dn_s=short['roi_image'].astype(np.float32)-args.black_level
    dn_l=long['roi_image'].astype(np.float32)-args.black_level
    v_s=ref & (dn_s>=args.min_signal_dn) & (dn_s<thresholds(short,args))
    v_l=ref & (dn_l>=args.min_signal_dn) & (dn_l<thresholds(long,args))
    nbase,c,frac=counts_of_masks(ref,v_s,v_l)
    E_s,E_l=short['E'],long['E']
    hs=hist_irr(dn_s,ref,v_s,E_s,edges,nbase)
    hl=hist_irr(dn_l,ref,v_l,E_l,edges,nbase)
    center=np.sqrt(edges[:-1]*edges[1:])
    vs=(center>=args.min_signal_dn/E_s)&(center<thresholds(short,args)/E_s)
    vl=(center>=args.min_signal_dn/E_l)&(center<thresholds(long,args)/E_l)
    # Histogram selection is by irradiance-bin measurement validity, not paired pixels.
    hf=np.where(vs&vl,(hs+hl)/2,np.where(vs,hs,np.where(vl,hl,0.0)))
    contributors=vs.astype(int)+vl.astype(int)
    source=np.where(vs&vl,'average',np.where(vs,'short',np.where(vl,'long','unobserved')))
    hrows=[dict(bin=i,irr_lo=f'{edges[i]:.12g}',irr_hi=f'{edges[i+1]:.12g}',
                short_fraction=f'{hs[i]:.12g}',long_fraction=f'{hl[i]:.12g}',
                fused_fraction=f'{hf[i]:.12g}',contributors=int(contributors[i]),source=source[i])
           for i in range(len(hf))]
    save_csv(out_dir/'histogram_bins.csv',list(hrows[0]),hrows)
    np.savez_compressed(out_dir/'histogram_data.npz',edges=edges,short=hs,long=hl,fused=hf,
                        contributors=contributors,short_valid_bins=vs,long_valid_bins=vl)
    # Diagnostics of low-signal both-invalid: only for images with corresponding spatial pixels.
    both_invalid=ref & ~v_s & ~v_l
    both_dim=both_invalid & (dn_s<args.min_signal_dn)&(dn_l<args.min_signal_dn)
    both_sat=both_invalid & ((dn_s>=thresholds(short,args))|(dn_l>=thresholds(long,args)))
    low_dn=dn_l[both_dim]
    row={'long_file':long['name'],'short_file':short['name'],
         'short_time_us':short['t'],'long_time_us':long['t'],
         'short_gain':short['gain'],'long_gain':long['gain'],
         'ratio_E':float(E_l/E_s),'total_selected_pixels':nbase,
         'both_invalid_pct':100*frac[0],'short_only_pct':100*frac[1],
         'long_only_pct':100*frac[2],'both_valid_pct':100*frac[3],
         'union_valid_pct':100*(1-frac[0]),
         'both_dim_pct':100*both_dim.sum()/nbase,
         'both_invalid_with_saturation_pct':100*both_sat.sum()/nbase,
         'both_dim_long_dn_p50':float(np.percentile(low_dn,50)) if len(low_dn) else None,
         'both_dim_long_dn_p90':float(np.percentile(low_dn,90)) if len(low_dn) else None,
         'both_dim_long_dn_p99':float(np.percentile(low_dn,99)) if len(low_dn) else None,
         'short_valid_pct':100*v_s.sum()/nbase,'long_valid_pct':100*v_l.sum()/nbase,
         'hist_fused_measured_mass':float(hf.sum()),
         'hist_short_measured_mass':float(hs.sum()),
         'hist_long_measured_mass':float(hl.sum())}
    (out_dir/'pair_summary.json').write_text(json.dumps(row,indent=2),encoding='utf8')
    # Composite diagnostic panel: histogram is independent of paired mask; mask assumes alignment.
    fig,ax=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
    ax[0,0].stairs(hs,edges,color='#2774ae',label='Short',lw=1.5)
    ax[0,0].stairs(hl,edges,color='#dd8833',label='Long',lw=1.5)
    ax[0,0].stairs(hf,edges,color='#222222',label='Fused',lw=2)
    ax[0,0].set_xscale('log');ax[0,0].set(xlabel='Relative irradiance DN/(us * gain)',ylabel='Fraction of ALL selected pixels/bin',title='Validity-aware histogram fusion')
    ax[0,0].legend();ax[0,0].grid(alpha=.2)
    mask=np.full(ref.shape,0,dtype=np.uint8)
    mask[ref & v_s & ~v_l]=1
    mask[ref & ~v_s & v_l]=2
    mask[ref & v_s & v_l]=3
    cm=ListedColormap(['#202020','#4699d2','#e5a14b','#80c78b'])
    ax[0,1].imshow(mask,cmap=cm,norm=BoundaryNorm(np.arange(-.5,4.5,1),cm.N),interpolation='nearest')
    ax[0,1].set_title(f'Pixel categories (assumes aligned views) | union={row["union_valid_pct"]:.1f}%')
    ax[0,1].set_xlabel('X pixel');ax[0,1].set_ylabel('Y pixel')
    ax[1,0].bar(['Both invalid','Short only','Long only','Both valid'],100*frac,
                 color=['#666666','#4699d2','#e5a14b','#80c78b'])
    ax[1,0].set(ylabel='Percentage of selected pixels (%)',ylim=(0,100),title='Coverage categories')
    for i,v in enumerate(frac):ax[1,0].text(i,v*100+1,f'{v*100:.1f}%',ha='center',fontsize=10)
    vals=dn_l[both_invalid]
    if len(vals):
        ax[1,1].hist(vals,bins=np.arange(0,max(40,int(args.min_signal_dn*3))+2,1),
                     color='#666666',alpha=.8)
    ax[1,1].axvline(args.min_signal_dn,color='tomato',linestyle='--',label=f'Low DN threshold={args.min_signal_dn:g}')
    ax[1,1].set(xlabel='Long RAW signal DN',ylabel='Both-invalid pixel count',title='Long DN distribution of both-invalid pixels')
    ax[1,1].legend(fontsize=9)
    fig.suptitle(f'Short: {short["name"]} | Long: {long["name"]} | Ratio={E_l/E_s:.2f}x')
    fig.savefig(out_dir/'01_pair_fusion_diagnostics.png',dpi=145);plt.close(fig)
    # Optional true pixel-based diagnostic only; requires aligned images.
    if args.save_masks:
        np.savez_compressed(out_dir/'valid_masks.npz',reference=ref,short=v_s,long=v_l,
                            both_invalid=both_invalid,both_dim=both_dim,union=v_s|v_l)
    return row


def sweep_plot(rows,out_dir):
    rows=sorted(rows,key=lambda r:r['ratio_E'])
    x=np.arange(len(rows));labels=[f'{r["ratio_E"]:.2f}x\n{r["long_time_us"]}us\n{r["long_gain"]:g}x' for r in rows]
    fig,(a,b)=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
    bottoms=np.zeros(len(rows))
    for key,label,color in [('both_invalid_pct','Both invalid','#666666'),('short_only_pct','Short only','#4699d2'),
                            ('long_only_pct','Long only','#e5a14b'),('both_valid_pct','Both valid','#80c78b')]:
        yy=np.array([r[key] for r in rows]);a.bar(x,yy,bottom=bottoms,label=label,color=color)
        bottoms+=yy
    a.set(xticks=x,xticklabels=labels,ylim=(0,100),ylabel='Fraction of selected pixels (%)',title='Pixel coverage by Long exposure')
    a.legend(loc='upper right',fontsize=9)
    for key,label,color in [('union_valid_pct','Union valid','#287b4c'),('long_only_pct','Long only','#dd8833'),
                            ('both_invalid_pct','Both invalid','#777777')]:
        b.plot(x,[r[key] for r in rows],'o-',label=label,color=color)
    b.set(xticks=x,xticklabels=labels,ylim=(0,100),ylabel='Fraction of selected pixels (%)',title='Coverage trend')
    b.grid(alpha=.25);b.legend()
    fig.savefig(out_dir/'00_coverage_sweep.png',dpi=165);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--input-dir',required=True,type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--short-file',required=True)
    p.add_argument('--long-files',nargs='*',default=[],help='Specific long RAW filenames; otherwise all greater-E files')
    p.add_argument('--same-gain-only',action='store_true',help='Filter longs to the same gain as short')
    p.add_argument('--gain-source',choices=['filename','header'],default='filename')
    p.add_argument('--gain-divisor',type=float,default=32.)
    p.add_argument('--black-level',type=float,default=0.)
    p.add_argument('--white-level',type=float,default=None)
    p.add_argument('--min-signal-dn',type=float,default=16.)
    p.add_argument('--sat-margin-dn',type=float,default=2.)
    p.add_argument('--bayer-pattern',choices=['RGGB','BGGR','GBRG','GRBG'],default='BGGR')
    p.add_argument('--channel',choices=['G','R','B','ALL'],default='G')
    p.add_argument('--roi',type=int,nargs=4,metavar=('X','Y','W','H'))
    p.add_argument('--bins',type=int,default=256)
    p.add_argument('--save-masks',action='store_true')
    args=p.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    short=parse_frame(args.input_dir/args.short_file,args)
    candidates=args.long_files or [p.name for p in sorted(args.input_dir.glob('*_bayer.pgm')) if p.name!=args.short_file]
    longs=[]
    for fname in candidates:
        f=parse_frame(args.input_dir/fname,args)
        if f['E']<=short['E']:
            warnings.warn(f'Skip non-long exposure: {fname}');continue
        if args.same_gain_only and not np.isclose(f['gain'],short['gain']):continue
        if f['image'].shape!=short['image'].shape:raise ValueError(f'Image shape mismatch: {fname}')
        longs.append(f)
    longs.sort(key=lambda f:f['E'])
    if not longs:raise ValueError('No eligible Long files found')
    # Identical ROI / CFA mask for all images (same sensor geometry). No alignment applied.
    sample,rx,ry=data_roi(short['image'],args.roi)
    ref=channel_mask(sample.shape,args.bayer_pattern,args.channel,rx,ry)
    short['roi_image']=sample
    for f in longs:f['roi_image']=data_roi(f['image'],args.roi)[0]
    # Global common log-bin edges; evaluate validity ranges for every Long on the same axis.
    allframes=[short]+longs
    irr_lo=min(args.min_signal_dn/f['E'] for f in allframes)
    irr_hi=max(thresholds(f,args)/f['E'] for f in allframes)
    if irr_lo<=0:raise ValueError('min-signal-dn must be positive for log histogram')
    edges=np.geomspace(irr_lo,irr_hi*(1+1e-10),args.bins+1)
    rows=[]
    for i,f in enumerate(longs):
        folder=args.output_dir/f'pair_{i+1:02d}_{f["name"].replace("_bayer.pgm", "")}'
        row=pair_analysis(short,f,ref,args,edges,folder)
        rows.append(row)
        print(f'{f["name"]}: R={row["ratio_E"]:.2f}x, both-invalid={row["both_invalid_pct"]:.1f}%, '
              f'long-only={row["long_only_pct"]:.1f}%, union={row["union_valid_pct"]:.1f}%, '
              f'fusion mass={row["hist_fused_measured_mass"]:.3f}')
    save_csv(args.output_dir/'coverage_sweep.csv',list(rows[0].keys()),rows)
    sweep_plot(rows,args.output_dir)
    (args.output_dir/'run_config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2),encoding='utf8')
    print('Output folder:',args.output_dir)
    print('IMPORTANT: pixel categories require aligned images; histogram fusion does not.')
    print('IMPORTANT: histogram mass is NOT a probability normalized to 1; report missing ranges separately.')

if __name__=='__main__':
  main()
