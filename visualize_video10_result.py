import h5py
from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import numpy as np

#parser = argparse.ArgumentParser()
#parser.add_argument('-p', '--path', type=str, required=True,
#                    help="path to h5 file containing summarization results")
#args = parser.parse_args()
path = "log/tvsum-split/result_user_summary_split_4.h5"
h5_res = h5py.File(path, 'r')
keys = h5_res.keys()
print(h5_res['video_10'].keys())

#### src: https://github.com/colizoli/xcorr_python
def xcorr(x, y, normed=True, detrend=False, maxlags=10):
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')
    
    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x)) # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))
    
    c = np.correlate(x, y, mode='full')

    if normed:
        n = np.sqrt(np.dot(x, x) * np.dot(y, y)) # this is the transformation function
        c = np.true_divide(c,n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c

for key in keys:
    if key == 'video_10':
        print("key = ",key)
        print("--------")
        score = h5_res[key]['score'][...]
        machine_summary = h5_res[key]['machine_summary'][...]
        gtscore = h5_res[key]['gtscore'][...]
        fm = h5_res[key]['fm'][()]

        # normlize value to [0, 1]
        norm_gtscore = (gtscore-min(gtscore)) / (max(gtscore)-min(gtscore))
        norm_score = (score-min(score)) / (max(score)-min(score))

        #print(max(norm_score))

        lags, c = xcorr(gtscore, score)
        max_xcorr = max(c) * 100

        #xcorr = np.correlate(norm_gtscore, norm_score)
        #print(xcorr)

        # plot score vs gtscore
        fig, ax1 = plt.subplots(figsize=(10, 4))
        n = len(gtscore)
        plt.title("Dataset=tvsum,canonical  Split=5  Video={}  F-score={:.1f}  XCorr={:.2f}".format(key, fm*100, max_xcorr))
        lns1 = ax1.plot(range(n), norm_gtscore, color='red', label="Ground Truth Summary")
        ax1.set_xlabel('Frame index (subsampled)')
        ax1.set_ylabel('Frame Importance Score')
        ax1.set_xlim(0, n)
        #ax1.set_ylim(0, 1.0)

        ax2 = ax1.twinx()
        ax2.set_yticklabels([])

        lns2 = ax2.plot(range(n), norm_score, color='blue', label="User Summary")
        # added these three lines
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc="upper left")

        fig.tight_layout()
        #plt.tick_params(axis='x',top=False)
        fig.savefig(os.path.join("results/charts/", 'score_' + key + '.png'), bbox_inches='tight')
        plt.show()
        plt.close()

        print("Index: ", [i for i, j in enumerate(machine_summary) if j > 0])
        print("Done video {}. # frames {}.".format(key, len(machine_summary)))


h5_res.close()