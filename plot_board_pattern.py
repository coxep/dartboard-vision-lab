"""Plot retained calibration evidence; no detector inputs are changed here."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    out=Path('pattern_evidence')
    seq=json.loads((out/'sequence.json').read_text())
    acquisition=json.loads((out/'holdout/acquisition.json').read_text())
    plt.style.use('dark_background')
    fig,axes=plt.subplots(2,2,figsize=(13,8.5),constrained_layout=True)
    fig.patch.set_facecolor('#141718')
    for ax in axes.flat:
        ax.set_facecolor('#141718');ax.grid(alpha=.15)
    for c,col in enumerate(('#53d8ae','#66b9ff','#f1c76b')):
        rows=[r for r in seq['frames'] if r['camera']==c and r['found']]
        axes[0,0].plot([r['frame'] for r in rows],[r['probe_max_px'] for r in rows],'.-',ms=3,color=col,label=f'Camera {c+1}')
        axes[1,0].plot([r['frame'] for r in rows],[r['camera_center_error_mm'] for r in rows],'.-',ms=3,color=col)
    for ax in (axes[0,0],axes[1,0]):
        ax.axvspan(20,29,color='#bbb',alpha=.1,label='Flat partial occlusion' if ax is axes[0,0] else None)
        ax.axvspan(40,44,color='#ef7373',alpha=.18,label='Board absent' if ax is axes[0,0] else None)
        ax.axvline(45,color='#ddd',ls=':',alpha=.7)
        ax.set_xlabel('Synthetic frame; camera jump at 45')
    axes[0,0].set(title='Fresh grayscale tracking: worst probe error',ylabel='Canonical pixels (900 × 900)',ylim=(0,1.5))
    axes[0,0].legend(fontsize=8,ncol=2)
    axes[1,0].set(title='Recovered physical camera position',ylabel='Camera-centre error (mm)',ylim=(0,1.5))
    good=[r for r in acquisition['views'] if r['found']]
    x=[r['same_points_free_H']['probe_median_px'] for r in good]
    y=[r['probe_median_px'] for r in good]
    axes[0,1].scatter(x,y,s=12,color='#53d8ae',alpha=.55)
    top=max(max(x),max(y))*1.05
    axes[0,1].plot([0,top],[0,top],color='#bbb',lw=1)
    axes[0,1].set(title='Same detected matches: physical pose vs free H',xlabel='Free-H median probe error (px)',ylabel='PnP median probe error (px)',xlim=(0,top),ylim=(0,top))
    bins=np.linspace(0,1.1,28)
    axes[1,1].hist([r['camera_center_error_mm'] for r in good],bins=bins,color='#66b9ff')
    axes[1,1].set(title=f"Cold acquisition: {len(good)}/300 views accepted; 3 retained rejections",xlabel='Camera-centre error (mm)',ylabel='Accepted views')
    fig.suptitle('Measured image correspondences → metric PnP → pose-derived rectification',fontsize=16)
    fig.savefig(out/'accuracy.png',dpi=160)
    plt.close(fig)


if __name__=='__main__':main()
