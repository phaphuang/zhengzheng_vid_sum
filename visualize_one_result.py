import h5py
from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import seaborn as sns
import numpy as np
#parser = argparse.ArgumentParser()
#parser.add_argument('-p', '--path', type=str, required=True,
#                    help="path to h5 file containing summarization results")
#args = parser.parse_args()
path = "log/tvsum-split/result_user_summary_split_4.h5"
h5_res = h5py.File(path, 'r')
keys = h5_res.keys()
print(h5_res['video_10'].keys())
c= 0
for key in keys:
    if key == 'video_10':
        print("key = ",key)
        print("--------")
        score = h5_res[key]['score'][...]
        machine_summary = h5_res[key]['machine_summary'][...]
        gtscore = h5_res[key]['gtscore'][...]
        fm = h5_res[key]['fm'][()]

        gt_frame_score = h5_res[key]['gt_frame_score'][...]

        user_summary = h5_res[key]['user_summary'][1]
    
        # plot score vs gtscore
        fig, axs = plt.subplots(2)
        n = len(gtscore)
        axs[0].plot(range(n), gtscore, color='red')
        axs[0].set_xlim(0, n)
        axs[0].set_yticklabels([])
        axs[0].set_xticklabels([])
        axs[1].set_title("video {} F-score {:.1%}".format(key, fm))
        axs[1].plot(range(n), score, color='blue')
        axs[1].set_xlim(0, n)
        axs[1].set_yticklabels([])
        axs[1].set_xticklabels([])
        #fig.savefig(osp.join(osp.dirname(args.path), 'score_' + key + '.png'), bbox_inches='tight')
        plt.show()
        plt.close()

        print("Done video {}. # frames {}.".format(key, len(machine_summary)))


        ############ Plot bar chart ############
        #### src: https://github.com/weirme/Video_Summary_using_FCSN/blob/master/gen_summary.py
        #print([i > 0 for i in machine_summary])
        fig = plt.figure(figsize=(10, 4))
        #print(gtscore)
        #print(score)
        sns.set()
        plt.bar(x=list(range(len(gt_frame_score))), height=gt_frame_score, color=['lightgray' if i == 0 else 'orange' for i in machine_summary], edgecolor=None, linewidth=0, width=1.0)
        plt.title(key)
        plt.tight_layout()
        plt.show()
        fig.savefig(os.path.join("results/charts/", 'machine_split_4_' + key + '.png'), bbox_inches='tight')
        plt.close()


h5_res.close()