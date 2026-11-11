#!/usr/bin/env python3
"""Render the published paired scoring figures from every retained trial."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    root=Path('detection_evidence')
    reports=[json.loads((root/f/'report.json').read_text()) for f in ('randomized','stress_holdout')]
    fig,axes=plt.subplots(1,2,figsize=(12,4.4),layout='constrained')
    colors={'correct':'#73bc85','wrong':'#ed795e','rejected':'#b7bdc6','calibration_failed':'#797f8e'}
    for ax,r,title in zip(axes,reports,['Original 100 synthetic throws','100-throw stress holdout']):
        modes=['legacy_estimated','new_estimated','new_oracle']
        names=['Old detector / estimated H','New detector / estimated H','New detector / exact H diagnostic']
        for i,mode in enumerate(modes):
            left=0
            for outcome,color in colors.items():
                count=r['summary'][mode]['counts'].get(outcome,0)
                ax.barh(i,count,left=left,color=color,label=outcome.replace('_',' ') if i==0 else None)
                if count>=3:ax.text(left+count/2,i,str(count),ha='center',va='center',fontsize=10)
                left+=count
        ax.set_yticks(range(3),names,fontsize=9);ax.invert_yaxis();ax.set_xlim(0,100)
        ax.set_xlabel('All throws, including failures');ax.set_title(title)
        ax.spines[['top','right']].set_visible(False)
    axes[0].legend(loc='lower left',bbox_to_anchor=(0,-.44),ncol=2,fontsize=9)
    fig.suptitle('Paired dart-detection comparison — synthetic evidence only',fontsize=14)
    fig.savefig(root/'comparison.png',dpi=170,bbox_inches='tight')
    fig.savefig(root/'comparison.svg',bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,r,title in zip(axes,reports,['Original randomized set','Stress holdout']):
        for mode,label,color in [('legacy_estimated','Old / estimated H','#888888'),
                                 ('new_estimated','New / estimated H','#007c67'),
                                 ('new_oracle','New / exact H diagnostic','#6d50b0')]:
            errors=sorted(row['results'][mode]['error_mm'] for row in r['trials'] if row['results'][mode]['error_mm'] is not None)
            ax.plot(errors,np.arange(1,len(errors)+1)/len(errors),label=label,color=color)
        ax.set_title(title);ax.set_xlabel('Accepted-hit position error (mm)')
        ax.set_ylabel('Fraction of accepted estimates');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Error distributions include wrong scores; accepted estimates only')
    fig.savefig(root/'errors.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()
