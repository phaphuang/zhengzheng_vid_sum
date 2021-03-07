import h5py
from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
#parser = argparse.ArgumentParser()
#parser.add_argument('-p', '--path', type=str, required=True,
#                    help="path to h5 file containing summarization results")
#args = parser.parse_args()
path = "log/tvsum-split/result_user_summary_split_1.h5"
h5_res = h5py.File(path, 'r')
keys = h5_res.keys()
print(h5_res['video_10'].keys())
c= 0
for key in keys:
    if key == 'video_11':
        print("key = ",key)
        print("--------")
        score = h5_res[key]['score'][...]
        machine_summary = h5_res[key]['machine_summary'][...]
        gtscore = h5_res[key]['gtscore'][...]
        fm = h5_res[key]['fm'][()]
    #    user_summary = h5_res[key]['user_summary'][...]
        user_summary = h5_res[key]['user_summary'][1]
        print("machine_summary",end=':')
        print(len(machine_summary))
        print("user_summary:",end=' ')
        print(len(h5_res[key]['user_summary'][1]))
        # plot score vs gtscore
        fig, axs = plt.subplots(nrows=2,ncols=1,constrained_layout=True)
        n = len(user_summary)
        axs[0].plot(range(n), user_summary, color='red')
        axs[0].set_xlim(0, n)
        axs[0].set_xlabel("ground truth frame")
        axs[0].yaxis.set_major_locator(MultipleLocator(1))
        axs[0].set_ylabel("select or not")
        axs[1].plot(range(n), machine_summary, color='blue')
        axs[1].set_xlim(0, n)
        axs[1].yaxis.set_major_locator(MultipleLocator(1))
        axs[1].set_xlabel("video frame")
        axs[1].set_ylabel("select or not")
        plt.show()
        #fig.savefig(osp.join(osp.dirname(path), 'score_' + key + '.png'), bbox_inches='tight')
        plt.close()
        
        print ("Done video {}. # frames {}.".format(key, len(machine_summary)))

h5_res.close()