#!/usr/bin/env python3
"""Experimental 2-exposure HDR EPR scoring from simulated uniform 60-bin AE histograms.

Standalone: pip install numpy matplotlib

Inputs are black-level-corrected (or specify --black-level) Bayer P5 PGM frames.
The two source frames estimate a common scene irradiance distribution. The same
estimated distribution is then used to score ALL candidate LE gains. This avoids
unfairly comparing different candidates using different source histograms.

IMPORTANT:
- The 60-bin simulator uses UNIFORM RAW DN bins; verify actual AE hardware bins.
- Each histogram bin is assumed uniform inside its DN interval when rebinned.
- Saturated values and values below reliable RAW measurement remain uncertain.
- The calibrated alpha values provided by the user are used literally;
  verify Y and variance are in the same units before treating scores as physical SNR.
- Score is an experimental utility, not the original paper's exact objective.
"""
import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ALPHA_TABLE = {32:105.0, 64:147.0, 128:195.0, 256:256.0, 512:361.0}


def read_pgm(path):
    # P5 parser with comments in header and 16-bit big-endian samples per Netpbm.
    with open(path, 'rb') as f:
        def token():
            while True:
                b = f.read(1)
                if not b:
                    raise ValueError('Unexpected EOF in PGM header')
                if b == b'#':
                    f.readline()
                elif b in b' \t\r\n':
                    continue
                else:
                    break
            buf=bytearray(b)
            while True:
                b=f.read(1)
                if not b or b in b' \t\r\n':
                    if b == b'\r':
                        pos=f.tell()
                        if f.read(1)!=b'\n': f.seek(pos)
                    break
                buf.extend(b)
            return buf.decode('ascii')
        if token() != 'P5': raise ValueError(f'Not binary P5 PGM: {path}')
        w,h,m = int(token()),int(token()),int(token())
        if not 1 <= m <= 65535: raise ValueError('Bad PGM maxval')
        dtype = np.uint8 if m < 256 else np.dtype('>u2')
        count=w*h*np.dtype(dtype).itemsize
        data=f.read(count)
        if len(data)!=count: raise ValueError(f'Truncated PGM: {path}')
    return np.frombuffer(data,dtype=dtype).reshape(h,w).astype(np.float32),m


def parse_name(name):
    m=re.fullmatch(r'(\d{3})(\d{5})_bayer\.pgm',name,re.I)
    if not m: raise ValueError(f'Expected GGGTTTTT_bayer.pgm: {name}')
    return int(m[1]),int(m[2])


def channel_values(img, channel, pattern):
    if channel == 'ALL': return img.ravel()
    cfa=np.array(list(pattern)).reshape(2,2)
    parts=[img[y::2,x::2].ravel() for y in range(2) for x in range(2) if cfa[y,x]==channel]
    return np.concatenate(parts)


def histogram_60(img, channel, pattern, black, dn_max, count):
    values=np.clip(channel_values(img,channel,pattern).astype(np.float64)-black,0,dn_max)
    edges=np.linspace(0,dn_max+1,count+1)
    counts=np.histogram(values,bins=edges)[0]
    return edges,counts/counts.sum(),counts,values


def alpha(g):
    gs=np.array(sorted(ALPHA_TABLE),dtype=float)
    if g<gs[0] or g>gs[-1]: raise ValueError(f'Gain {g} outside alpha table')
    return float(np.interp(g,gs,[ALPHA_TABLE[int(x)] for x in gs]))


def snr2(y, gain, beta, alpha_scale=1.0):
    y=np.maximum(y,0)
    return y*y/np.maximum(alpha(gain)*alpha_scale*y+beta,1e-12)


def reliability(y,gain,beta,white,alpha_scale=1.0,snr_soft_scale=1.0):
    """Continuous informational reliability: weak signal and near-clipping penalty.

    SNR^2/(SNR^2 + scale) is continuous, not a hard lower DN cutoff.
    Clipping taper is logistic with width proportional to sensor noise std.
    These are modeling choices and need validation, not physical probabilities.
    """
    s2=snr2(y,gain,beta,alpha_scale)
    r_low=s2/(s2+snr_soft_scale)
    sigma=np.sqrt(np.maximum(alpha(gain)*alpha_scale*np.maximum(y,0)+beta,1e-12))
    width=np.maximum(sigma,1.0)
    z=np.clip((y-white)/width,-60,60)
    r_high=np.where(y>=white, 0.0, 1/(1+np.exp(z)))
    return r_low*r_high


def convert_hist_to_scene(dn_edges,probs,E,irr_edges,method):
    """Convert 60 DN bins to common irradiance bins, with fractional interval overlap.

    method='uniform': assume uniform signal within original DN interval.
    method='center': put each DN bin mass at its midpoint (quantization stress test).
    """
    if method == 'center':
        vals=(dn_edges[:-1]+dn_edges[1:])/(2*E)
        return np.histogram(vals,bins=irr_edges,weights=probs)[0]
    out=np.zeros(len(irr_edges)-1,dtype=np.float64)
    for j,p in enumerate(probs):
        if p==0:continue
        lo,hi=dn_edges[j]/E,dn_edges[j+1]/E
        if hi<=lo:continue
        start=max(0,int(np.searchsorted(irr_edges,lo,side='right')-1))
        stop=min(len(out),int(np.searchsorted(irr_edges,hi,side='left')+1))
        for k in range(start,stop):
            overlap=max(0.0,min(hi,irr_edges[k+1])-max(lo,irr_edges[k]))
            if overlap>0:out[k]+=p*overlap/(hi-lo)
    return out


def estimate_scene(dn_edges,p_se,p_le,tse,gse,tle,gle,irr_edges,
                   beta,white,alpha_scale,snr_soft_scale,method):
    E_se=tse*gse/32;E_le=tle*gle/32
    a=convert_hist_to_scene(dn_edges,p_se,E_se,irr_edges,method)
    b=convert_hist_to_scene(dn_edges,p_le,E_le,irr_edges,method)
    center=np.sqrt(irr_edges[:-1]*irr_edges[1:])
    w_se=reliability(center*E_se,gse,beta,white,alpha_scale,snr_soft_scale)
    w_le=reliability(center*E_le,gle,beta,white,alpha_scale,snr_soft_scale)
    # Blend independent frame estimates in the same irradiance domain. When
    # reliability is extremely low, information is reported as unobserved;
    # never renormalize 'known' mass to pretend all of the scene is measured.
    denom=w_se+w_le
    scene=np.divide(w_se*a+w_le*b,denom,out=np.zeros_like(a),where=denom>1e-12)
    confidence=np.maximum(w_se,w_le)
    # Down-weight unknown irradiance mass in score without deleting it silently.
    # 'scene' is a statistical estimate with no guarantee of mass conservation.
    return scene,center,a,b,w_se,w_le,confidence


def score_candidates(scene,center,candidates,tse,gse,tle,beta,white,
                     alpha_scale,snr_soft_scale):
    e_se=tse*gse/32
    y_se=center*e_se
    w_se=reliability(y_se,gse,beta,white,alpha_scale,snr_soft_scale)
    s2_se=snr2(y_se,gse,beta,alpha_scale)
    denom=max(float(scene.sum()),1e-12)
    out=[]
    surfaces=[]
    for g in candidates:
        e_le=tle*g/32
        y_le=center*e_le
        w_le=reliability(y_le,g,beta,white,alpha_scale,snr_soft_scale)
        s2_le=snr2(y_le,g,beta,alpha_scale)
        merged=w_se*s2_se+w_le*s2_le
        utility=np.log1p(merged)
        # Per-scene-pixel expected utility, not normalized by known mass.
        score=float(np.sum(scene*utility))
        # Separate metrics used to understand score behavior.
        short_sat=r_high(y_se,gse,beta,white,alpha_scale)
        long_sat=r_high(y_le,g,beta,white,alpha_scale)
        both_saturated=float(np.sum(scene*(1-short_sat)*(1-long_sat))/denom)
        le_clipped=float(np.sum(scene*(1-long_sat))/denom)
        quality=float(np.sum(scene*(1-np.exp(-merged)))/denom)
        out.append(dict(gain_code=g,le_time_us=tle,ratio=e_le/e_se,
                        score=score,measured_scene_mass=float(scene.sum()),
                        mean_smooth_snr_coverage=quality,
                        long_clip_risk=le_clipped,both_clip_risk=both_saturated,
                        mean_snr2=float(np.sum(scene*merged)/denom)))
        surfaces.append(utility)
    return out,np.array(surfaces)


def r_high(y,gain,beta,white,alpha_scale):
    sig=np.sqrt(np.maximum(alpha(gain)*alpha_scale*np.maximum(y,0)+beta,1e-12))
    z=np.clip((y-white)/np.maximum(sig,1),-60,60)
    return np.where(y>=white, 0.0, 1/(1+np.exp(z)))


def save_csv(path,rows):
    with open(path,'w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def plot_all(output,dn_edges,p_se,p_le,irr_edges,scene,center,a,b,confidence,
             rows,surface,metadata,sensitivity):
    x=np.arange(len(p_se))
    ratios=np.array([z['ratio'] for z in rows]);scores=np.array([z['score'] for z in rows])
    fig,axs=plt.subplots(2,2,figsize=(15,10),constrained_layout=True)
    ax=axs[0,0];ax.plot(x,p_se,'o-',ms=2.5,label='SE')
    ax.plot(x,p_le,'o-',ms=2.5,label='LE')
    ax.set(xlabel='Uniform AE bin index',ylabel='Fraction of pixels/bin',title='Source 60-bin histograms');ax.legend();ax.grid(alpha=.25)
    ax=axs[0,1]
    ax.stairs(a,irr_edges,label='SE mapped',color='#4386bf',lw=1)
    ax.stairs(b,irr_edges,label='LE mapped',color='#e39d43',lw=1)
    ax.stairs(scene,irr_edges,label='Scene estimate',color='black',lw=1.4)
    ax.set_xscale('log');ax.set(xlabel='Relative irradiance (DN/us at gain 1x)',ylabel='Pixel fraction / bin',title='Common scene irradiance estimate');ax.legend();ax.grid(alpha=.2)
    ax=axs[1,0]
    ax.plot(ratios,scores,'o-',lw=2,label='Primary SNR utility score')
    best=np.argmax(scores);ax.axvline(ratios[best],color='tab:green',ls='--',label=f'Best {ratios[best]:.2f}x')
    ax.set_xscale('log',base=2);ax.set_xticks(ratios,[f'{r:.2f}x' for r in ratios]);ax.tick_params(axis='x',rotation=25)
    ax.set(xlabel='LE / SE signal exposure ratio',ylabel='Histogram weighted log(1+SNR²)',title='EPR score curve');ax.grid(alpha=.3);ax.legend(fontsize=9)
    ax=axs[1,1]
    ax.plot(ratios,[r['long_clip_risk'] for r in rows],'o-',label='LE soft clipping risk')
    ax.plot(ratios,[r['both_clip_risk'] for r in rows],'o-',label='Both exposures clip risk')
    ax.plot(ratios,[r['mean_smooth_snr_coverage'] for r in rows],'o-',label='Soft SNR quality')
    ax.set_xscale('log',base=2);ax.set_xticks(ratios,[f'{r:.2f}x' for r in ratios]);ax.tick_params(axis='x',rotation=25)
    ax.set(xlabel='LE / SE signal exposure ratio',ylabel='Weighted index (0–1)',title='Score diagnostics (not hard thresholds)');ax.legend(fontsize=9);ax.grid(alpha=.3)
    fig.suptitle('2-Exposure HDR ratio scoring | 60-bin AE histogram (experimental)')
    fig.savefig(output/'01_epr_scoring_dashboard.png',dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(11,5),constrained_layout=True)
    ss=[]
    for label,results in sensitivity.items():
        ys=[r['score'] for r in results]
        ax.plot(ratios,ys,'o-',label=label);ss.append(ys)
    ax.set_xscale('log',base=2);ax.set_xticks(ratios,[f'{r:.2f}x' for r in ratios]);ax.grid(alpha=.3)
    ax.set(xlabel='EPR',ylabel='Score (same arbitrary utility units)',title='Sensitivity to noise-model scale and 60-bin quantization')
    ax.legend(fontsize=9);fig.savefig(output/'02_score_sensitivity.png',dpi=160);plt.close(fig)

    fig,ax=plt.subplots(figsize=(12,5),constrained_layout=True)
    fig_im=ax.imshow(surface,aspect='auto',origin='lower',extent=[np.log10(center[0]),np.log10(center[-1]),-.5,len(rows)-.5],cmap='viridis')
    ax.set(yticks=np.arange(len(rows)),yticklabels=[f'{r:.2f}x | gain {g["gain_code"]}' for r,g in zip(ratios,rows)],
           xlabel='log10(relative scene irradiance)',ylabel='Candidate EPR',title='SNR utility per scene brightness bin')
    fig.colorbar(fig_im,ax=ax,label='log(1+weighted SNR²)')
    fig.savefig(output/'03_snr_utility_heatmap.png',dpi=150);plt.close(fig)


def main():
    ap=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--input-dir',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,required=True)
    ap.add_argument('--se-file',required=True)
    ap.add_argument('--le-file',required=True,help='An actual LE RAW for building a fixed scene histogram')
    ap.add_argument('--candidate-gains',type=int,nargs='+',default=[32,64,96,128,256,512])
    ap.add_argument('--candidate-le-time-us',type=int,default=33000)
    ap.add_argument('--gain-divisor',type=float,default=32)
    ap.add_argument('--black-level',type=float,default=0)
    ap.add_argument('--white-level',type=float,default=1023)
    ap.add_argument('--channel',choices=['G','R','B','ALL'],default='G')
    ap.add_argument('--bayer-pattern',choices=['BGGR','RGGB','GBRG','GRBG'],default='BGGR')
    ap.add_argument('--ae-bins',type=int,default=60)
    ap.add_argument('--scene-bins',type=int,default=256)
    ap.add_argument('--beta',type=float,default=9.0,help='Assumed read-noise variance (DN²)')
    ap.add_argument('--alpha-scale',type=float,default=1.0,help='1 means BNC alpha is used directly')
    ap.add_argument('--snr-soft-scale',type=float,default=1.0,help='Smooth SNR information weighting (no hard DN threshold)')
    args=ap.parse_args()
    if args.beta<=0 or args.alpha_scale<=0 or args.snr_soft_scale<=0:raise ValueError('Noise parameters must be >0')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    se_g,se_t=parse_name(args.se_file);le_g,le_t=parse_name(args.le_file)
    se_img,se_max=read_pgm(args.input_dir/args.se_file)
    le_img,le_max=read_pgm(args.input_dir/args.le_file)
    if se_img.shape!=le_img.shape:raise ValueError('Input frame sizes differ')
    dn_max=min(int(args.white_level),se_max,le_max)
    dn_edges,pse,cse,_=histogram_60(se_img,args.channel,args.bayer_pattern,args.black_level,dn_max,args.ae_bins)
    _,ple,cle,_=histogram_60(le_img,args.channel,args.bayer_pattern,args.black_level,dn_max,args.ae_bins)
    E_se=se_t*se_g/args.gain_divisor; E_le=le_t*le_g/args.gain_divisor
    # Minimum positive irradiance is bounded by quantization of the most exposed frame.
    lo=0.5/max(E_se,E_le)
    hi=(dn_max+1)/min(E_se,E_le)
    irr_edges=np.geomspace(lo,hi,args.scene_bins+1)
    # Censor first half-DN close to black; log axis otherwise cannot represent zero.
    # The mass below the smallest positive bin is explicitly lost and reported.
    scene,center,a,b,wse,wle,conf=estimate_scene(dn_edges,pse,ple,se_t,se_g,le_t,le_g,irr_edges,
        args.beta,dn_max,args.alpha_scale,args.snr_soft_scale,'uniform')
    # This heuristic blend is not guaranteed to conserve mass. Normalize it as an
    # estimated *conditional* scene distribution for candidate comparison.
    # Raw blended mass is diagnostic only, NOT a coverage estimate.
    unnormalized_mass=float(scene.sum())
    if unnormalized_mass <= 0: raise ValueError('No usable scene histogram information')
    scene=scene/unnormalized_mass
    candidates=sorted(set(args.candidate_gains),key=lambda g:args.candidate_le_time_us*g)
    rows,surface=score_candidates(scene,center,candidates,se_t,se_g,args.candidate_le_time_us,
                                  args.beta,dn_max,args.alpha_scale,args.snr_soft_scale)
    # Same scene estimate for all candidates, including sensitivity curves.
    sensitivity={'BNC alpha (base)':rows}
    for scale in [0.5,2.0]:
        r,_=score_candidates(scene,center,candidates,se_t,se_g,args.candidate_le_time_us,
             args.beta,dn_max,args.alpha_scale*scale,args.snr_soft_scale)
        sensitivity[f'alpha x {scale:g}']=r
    # Quantization sensitivity: replace within-bin uniform hypothesis by midpoints.
    scene2,*_=estimate_scene(dn_edges,pse,ple,se_t,se_g,le_t,le_g,irr_edges,
         args.beta,dn_max,args.alpha_scale,args.snr_soft_scale,'center')
    if scene2.sum()>0: scene2=scene2/scene2.sum()
    r2,_=score_candidates(scene2,center,candidates,se_t,se_g,args.candidate_le_time_us,
         args.beta,dn_max,args.alpha_scale,args.snr_soft_scale)
    sensitivity['60-bin center-point'] = r2
    save_csv(args.output_dir/'candidate_scores.csv',rows)
    np.savez_compressed(args.output_dir/'scene_estimate.npz',irr_edges=irr_edges,irr_centers=center,
            scene_hist=scene,se_hist=a,le_hist=b,se_weight=wse,le_weight=wle,
            ae_dn_edges=dn_edges,ae_se=pse,ae_le=ple,candidate_utility=surface)
    plot_all(args.output_dir,dn_edges,pse,ple,irr_edges,scene,center,a,b,conf,rows,surface,args,sensitivity)
    summary={'source_se':args.se_file,'source_le':args.le_file,'source_exposure_ratio':E_le/E_se,
             'pre_normalization_blended_mass':unnormalized_mass,
             'scene_hist_probability_sum':float(scene.sum()),
             'short_bin0_fraction':float(pse[0]),'long_bin0_fraction':float(ple[0]),
             'selected_best_candidate':max(rows,key=lambda x:x['score']),
             'note':'Scores depend on alpha DN units, assumed beta, 60-bin redistribution, and scene truncation. NOT ground truth.'}
    (args.output_dir/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    (args.output_dir/'config.json').write_text(json.dumps({k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},indent=2),encoding='utf8')
    print(f'Source SE: {args.se_file}, LE: {args.le_file} | source ratio {E_le/E_se:.3f}x')
    print(f'SE bin0 {100*pse[0]:.1f}% | LE bin0 {100*ple[0]:.1f}% | pre-normalization blended mass {unnormalized_mass:.3f} (NOT coverage)')
    for row in rows:
        print(f'EPR {row["ratio"]:6.2f}x | Gain {row["gain_code"]:3d} | Score {row["score"]:.6g} | LE clip risk {row["long_clip_risk"]:.3f} | quality {row["mean_smooth_snr_coverage"]:.3f}')
    best=max(rows,key=lambda r:r['score'])
    print(f'EXPERIMENTAL TOP RATIO {best["ratio"]:.2f}x (gain {best["gain_code"]})')
    print('Check 02_score_sensitivity.png before trusting the ranking.')
    print('Output:',args.output_dir)

if __name__=='__main__':main()
