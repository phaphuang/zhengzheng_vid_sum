from __future__ import print_function
import os
import os.path as osp
import argparse
import sys
import h5py
import time
import datetime
import numpy as np
from tabulate import tabulate

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.distributions import Bernoulli
from model_trans import *
from utils import Logger, read_json, write_json, save_checkpoint
from models import *
from rewards import compute_reward
import vsum_tools
from torch.autograd import Variable
import re
import random

from scores.eval import generate_scores, evaluate_scores

parser = argparse.ArgumentParser("Pytorch code for unsupervised video summarization with REINFORCE")
# Dataset options
parser.add_argument('-d', '--dataset', type=str, required=False, help="path to h5 dataset (required)")
parser.add_argument('-us', '--userscore', type=str, required=True, help="path to h5 of user's scores (required)")
parser.add_argument('-s', '--split', type=str, required=True, help="path to split file (required)")
parser.add_argument('--split-id', type=int, default=0, help="split index (default: 0)")
parser.add_argument('-m', '--metric', type=str, required=True, choices=['tvsum', 'summe'],
                    help="evaluation metric ['tvsum', 'summe']")
# Model options
parser.add_argument('--input-dim', type=int, default=1024, help="input dimension (default: 1024)")
parser.add_argument('--hidden-dim', type=int, default=256, help="hidden unit dimension of DSN (default: 256)")
parser.add_argument('--num-layers', type=int, default=1, help="number of RNN layers (default: 1)")
parser.add_argument('--rnn-cell', type=str, default='lstm', help="RNN cell type (default: lstm)")
# Optimization options
parser.add_argument('--lr', type=float, default=1e-05, help="learning rate (default: 1e-05)")
parser.add_argument('--attn_lr', type=float, default=1e-04, help="transformer learning rate (default: 2e-04)")
parser.add_argument('--meta_lr', type=float, default=1e-05, help="meta learning rate (default: 1e-05)")
parser.add_argument('--weight-decay', type=float, default=1e-05, help="weight decay rate (default: 1e-05)")
parser.add_argument('--bsz', type=int, default=2, help="batch size Multiple of 5 is better(default: 5)")
parser.add_argument('--max-epoch', type=int, default=10, help="maximum epoch for training (default: 60)")
parser.add_argument('--meta_step', type=int, default=1, help="meta epoch for training (default: 10)")
parser.add_argument('--stepsize', type=int, default=30, help="how many steps to decay learning rate (default: 30)")
parser.add_argument('--gamma', type=float, default=0.1, help="learning rate decay (default: 0.1)")
parser.add_argument('--num-episode', type=int, default=5, help="number of episodes (default: 5)")
parser.add_argument('--beta', type=float, default=0.01, help="weight for summary length penalty term (default: 0.01)")
# Misc
parser.add_argument('--seed', type=int, default=1, help="random seed (default: 1)")
parser.add_argument('--gpu', type=str, default='0', help="which gpu devices to use")
parser.add_argument('--use-cpu', action='store_true', help="use cpu device")
parser.add_argument('--evaluate', action='store_true', help="whether to do evaluation only")
parser.add_argument('--save-dir', type=str, default='log', help="path to save output (default: 'log/')")
parser.add_argument('--resume', type=str, default='', help="path to resume file")
parser.add_argument('--verbose', action='store_true', help="whether to show detailed test results")
parser.add_argument('--save-results', action='store_true', help="whether to save output results")
parser.add_argument('--attention', action='store_true',help="with/without attention")

#最好结果内1e-3，外1e-4
#原本外1e-4，内1e-5

args = parser.parse_args()
attention = args.attention
attn_lr = args.attn_lr #随便设置的
meta_lr = args.meta_lr
meta_step = args.meta_step
bsz = args.bsz # batch size
#input_dim = 1000
torch.manual_seed(args.seed)
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
use_gpu = torch.cuda.is_available()
if args.use_cpu: use_gpu = False

def main():
    
    if not args.evaluate:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_train.txt'))
    else:
        sys.stdout = Logger(osp.join(args.save_dir, 'log_test.txt'))
    
    print("==========\nArgs:{}\n==========".format(args))
    
    if use_gpu:
        print("Currently using GPU {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Initialize dataset {}".format(args.dataset))
    if args.dataset is None:
        datasets = ['datasets/eccv16_dataset_summe_google_pool5_pyramid_1024.h5',
                'datasets/eccv16_dataset_tvsum_google_pool5_pyramid_1024.h5',
                'datasets/eccv16_dataset_ovp_google_pool5.h5',
                'datasets/eccv16_dataset_youtube_google_pool5.h5']
        
        dataset = {}
        for name in datasets:
            _, base_filename = os.path.split(name)
            base_filename = os.path.splitext(base_filename)
            dataset[base_filename[0]] = h5py.File(name, 'r')
        # Load split file
        splits = read_json(args.split)
        assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
        split = splits[args.split_id]
        train_keys = split['train_keys']
        test_keys = split['test_keys']
        print("# train videos {}. # test videos {}".format(len(train_keys), len(test_keys)))
    
    else:
        dataset = h5py.File(args.dataset, 'r')
        num_videos = len(dataset.keys())
        splits = read_json(args.split)
        assert args.split_id < len(splits), "split_id (got {}) exceeds {}".format(args.split_id, len(splits))
        split = splits[args.split_id]
        train_keys = split['train_keys']
        test_keys = split['test_keys']
        print("# total videos {}. # train videos {}. # test videos {}".format(num_videos, len(train_keys), len(test_keys)))


    userscoreset = h5py.File(args.userscore, 'r')
    print("Initialize model")
    #model选型
    print("attention = ",attention)
    #print("bsz = ",bsz)
    #1.原始LSTM
    
    #2.transform模型，无LSTM
     #args.input_dim
    if attention==True: 
        attn_model = make_model(d_model = args.input_dim)
        attn_meta_model = make_model(d_model = args.input_dim)
        attn_optimizer = torch.optim.Adam(attn_model.parameters(), betas=(0.5,0.999),lr=attn_lr, weight_decay=args.weight_decay) #简单添加attn optimizer
        attn_meta_optimizer = torch.optim.Adam(attn_model.parameters(), betas=(0.5,0.999),lr=meta_lr, weight_decay=args.weight_decay) #简单添加attn optimizer
    else:
        model = MRN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
        meta_model = MRN(in_dim=args.input_dim, hid_dim=args.hidden_dim, num_layers=args.num_layers, cell=args.rnn_cell)
        optimizer = torch.optim.Adam(model.parameters(), betas=(0.5,0.999),lr=args.lr, weight_decay=args.weight_decay)
        meta_optimizer = torch.optim.Adam(model.parameters(), betas=(0.5,0.999),lr=meta_lr, weight_decay=args.weight_decay) #使用adam优化
    if use_gpu:
        if attention==True:
            attn_model = nn.DataParallel(attn_model).cuda()#给attn_model加cuda
            attn_meta_model = nn.DataParallel(attn_meta_model).cuda()
        else:
            model = nn.DataParallel(model).cuda()
            meta_model = nn.DataParallel(meta_model).cuda()
   #如果不能运行就把这个放到use_gpu后面，或者前面
    if args.resume:
        print("Loading checkpoint from '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        if attention == False:
            model.load_state_dict(checkpoint)
        else:
            attn_model.load_state_dict(checkpoint)
    else:
        start_epoch = 0
   
    if args.evaluate:
        print("Evaluate only")
        if attention==False:
            evaluate(model,dataset, test_keys, use_gpu)
        else:
            evaluate(attn_model,dataset, test_keys, use_gpu)
        return
    
    print("==> Start training")
    start_time = time.time()
    if attention == False:
        model.train()
        meta_model.train()
    else:
        attn_model.train()
        attn_meta_model.train()
    n = 0

    if args.dataset is None:
        while n < meta_step:
            #重设meta_model
            total_loss = 0
            total_times = 0
            baselines = {key: 0. for key in train_keys} # 基本reward
            reward_writers = {key: [] for key in train_keys} # 记录reward的变化
    #        if attention == False:
    #            model.load_state_dict(meta_model.state_dict())
    #        else: 
    #            attn_model.load_state_dict(attn_meta_model.state_dict())
            for epoch in range(start_epoch, args.max_epoch):
                epis_rewards = []
                b = 0
                training = train_keys.copy()
                while (len(training) != 0):
                    choose_list = random.sample(training,bsz)
                    #print('Batch------',b)
                    b = b+1
                    training = list(set(training)-set(choose_list))
                    for choose_path in choose_list:
                        key_parts = choose_path.split('/')
                        name, key = key_parts
                        #print(key,end=' ')
                        seq = dataset[name][key]['features'][...] # features
                        seq = torch.from_numpy(seq).unsqueeze(0) # 给features增加一维，改变维度为（1，seq_len,1000）
                        if use_gpu: seq = seq.cuda() 
                        if attention == True:
                            probs = attn_meta_model(seq.float()) #（1，seq_len,1）
                            #torch.cuda.empty_cache()
                        else:
                            probs = meta_model(seq) # 输出维度 (1, seq_len, 1)
                            #torch.cuda.empty_cache()
                        cost_1 = args.beta * (probs.mean() - 0.5)**2 # 最小化摘要长度惩罚因子损失函数
                        m1 = Bernoulli(probs)
                        
                        actions_1 = m1.sample() #采样
                        log_probs_1 = m1.log_prob(actions_1) #生成损失函数
                        reward_1 = compute_reward(seq, actions_1, use_gpu=use_gpu)
                        expected_reward_1 = log_probs_1.mean() * (reward_1 - baselines[choose_path])
                        cost_1 -= expected_reward_1 # 最小化负向期望reward
                        epis_rewards.append(reward_1.item())
    # =============================================================================
    #                         without attention
    # =============================================================================
                        if attention == False:
                            meta_optimizer.zero_grad()
                            cost_1.backward()
                            meta_optimizer.step()
                            
                            total_loss += cost_1.item() #综合损失
                            total_times += 1
                            baselines[choose_path] = 0.9 * baselines[choose_path] + 0.1 * np.mean(epis_rewards) # 更新reward
                            reward_writers[choose_path].append(np.mean(epis_rewards))
                        else:
    # =============================================================================
    #                         with attention
    # =============================================================================
                            attn_optimizer.zero_grad()
                            cost_1.backward()
                            attn_optimizer.step()
                            
                            total_loss += cost_1.item() #综合损失
                            total_times += 1
                            baselines[choose_path] = 0.9 * baselines[choose_path] + 0.1 * np.mean(epis_rewards) # 更新reward
                            reward_writers[choose_path].append(np.mean(epis_rewards))
                    #print("")
                epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
                print("epoch {}/{} meta_epoch {}/{}\t reward {}\t loss {} ".format(epoch+1, args.max_epoch,n+1,meta_step, epoch_reward,total_loss/total_times))
            if attention == True:
                cost_2 = torch.tensor(total_loss/total_times)
                cost_2 = Variable(cost_2,requires_grad=True)
                attn_optimizer.zero_grad()
                cost_2.backward()
                attn_optimizer.step()
            else:
                cost_2 = torch.tensor(total_loss/total_times)
                cost_2 = Variable(cost_2,requires_grad=True)
                optimizer.zero_grad()
                cost_2.backward()
                optimizer.step()
            n=n+1
    else:
        while n < meta_step:
            #重设meta_model
            total_loss = 0
            total_times = 0
            baselines = {key: 0. for key in train_keys} # 基本reward
            reward_writers = {key: [] for key in train_keys} # 记录reward的变化
    #        if attention == False:
    #            model.load_state_dict(meta_model.state_dict())
    #        else: 
    #            attn_model.load_state_dict(attn_meta_model.state_dict())
            for epoch in range(start_epoch, args.max_epoch):
                epis_rewards = []
                b = 0
                training = train_keys.copy()
                while (len(training) != 0):
                    choose_list = random.sample(training,bsz)
                    #print('Batch------',b)
                    b = b+1
                    training = list(set(training)-set(choose_list))
                    for idx in choose_list:
                        key = idx
                        #print(key,end=' ')
                        seq = dataset[key]['features'][...] # features
                        seq = torch.from_numpy(seq).unsqueeze(0) # 给features增加一维，改变维度为（1，seq_len,1000）
                        if use_gpu: seq = seq.cuda() 
                        if attention == True:
                            probs = attn_meta_model(seq) #（1，seq_len,1）
                            #torch.cuda.empty_cache()
                        else:
                            probs = meta_model(seq) # 输出维度 (1, seq_len, 1)
                            #torch.cuda.empty_cache()
                        cost_1 = args.beta * (probs.mean() - 0.5)**2 # 最小化摘要长度惩罚因子损失函数
                        m1 = Bernoulli(probs)
                        
                        actions_1 = m1.sample() #采样
                        log_probs_1 = m1.log_prob(actions_1) #生成损失函数
                        reward_1 = compute_reward(seq, actions_1, use_gpu=use_gpu)
                        expected_reward_1 = log_probs_1.mean() * (reward_1 - baselines[key])
                        cost_1 -= expected_reward_1 # 最小化负向期望reward
                        epis_rewards.append(reward_1.item())
    # =============================================================================
    #                         without attention
    # =============================================================================
                        if attention == False:
                            meta_optimizer.zero_grad()
                            cost_1.backward()
                            meta_optimizer.step()
                            
                            total_loss += cost_1.item() #综合损失
                            total_times += 1
                            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # 更新reward
                            reward_writers[key].append(np.mean(epis_rewards))
                        else:
    # =============================================================================
    #                         with attention
    # =============================================================================
                            attn_optimizer.zero_grad()
                            cost_1.backward()
                            attn_optimizer.step()
                            
                            total_loss += cost_1.item() #综合损失
                            total_times += 1
                            baselines[key] = 0.9 * baselines[key] + 0.1 * np.mean(epis_rewards) # 更新reward
                            reward_writers[key].append(np.mean(epis_rewards))
                    #print("")
                epoch_reward = np.mean([reward_writers[key][epoch] for key in train_keys])
                print("epoch {}/{} meta_epoch {}/{}\t reward {}\t loss {} ".format(epoch+1, args.max_epoch,n+1,meta_step, epoch_reward,total_loss/total_times))
            if attention == True:
                cost_2 = torch.tensor(total_loss/total_times)
                cost_2 = Variable(cost_2,requires_grad=True)
                attn_optimizer.zero_grad()
                cost_2.backward()
                attn_optimizer.step()
            else:
                cost_2 = torch.tensor(total_loss/total_times)
                cost_2 = Variable(cost_2,requires_grad=True)
                optimizer.zero_grad()
                cost_2.backward()
                optimizer.step()
            n=n+1
    write_json(reward_writers, osp.join(args.save_dir, 'rewards.json'))
    if attention == False:
        mean_fm = evaluate(model,dataset, userscoreset, test_keys, use_gpu)
        mean_fm = round(mean_fm,2)
        model_state_dict = model.state_dict()
    else:
        mean_fm = evaluate(attn_model,dataset, userscoreset, test_keys, use_gpu)
        mean_fm = round(mean_fm,2)
        model_state_dict = attn_model.state_dict()
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))
    model_save_path = osp.join(args.save_dir, ('attn_' if attention==True else '')+args.metric+'_FScore_'+str(mean_fm)+'_meta_epoch_'+str(meta_step)+'_inner_epoch_'+ str(args.max_epoch) +'_split_id_' +str(args.split_id)+'_feature_'+str(args.input_dim)+'.pth.tar')
    #save_checkpoint(model_state_dict, model_save_path)
    #print("Model saved to {}".format(model_save_path))
    #dataset.close()
    userscoreset.close()

def evaluate(model, dataset, userscoreset, test_keys, use_gpu):
    print("==> Test")
    #use_meta = False
    with torch.no_grad():
        model.eval()
        fms = []
        eval_metric = 'avg' if args.metric == 'tvsum' else 'max'
        #eval_metric = 'avg'
        if args.verbose: table = [["No.", "Video", "F-score"]]

        if args.save_results:
            h5_res = h5py.File(osp.join(args.save_dir, 'result_user_summary.h5'), 'w')

        spear_avg_corrs = []
        kendal_avg_corrs = []

        if args.dataset is None:
            for key_idx, key in enumerate(test_keys):
                key_parts = test_keys[key_idx].split('/')
                name, key = key_parts
                seq = dataset[name][key]['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                if use_gpu: seq = seq.cuda()
                probs = model(seq)
                probs = probs.data.cpu().squeeze().numpy()


                cps = dataset[name][key]['change_points'][...]
                num_frames = dataset[name][key]['n_frames'][()]
                nfps = dataset[name][key]['n_frame_per_seg'][...].tolist()
                positions = dataset[name][key]['picks'][...]
                user_summary = dataset[name][key]['user_summary'][...]

                gtscore = dataset[name][key]['gtscore'][...]

                #print(key, len(probs), len(gtscore))

                machine_summary, _ = vsum_tools.generate_summary(probs, gtscore, cps, num_frames, nfps, positions)
                fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
                fms.append(fm)

                #### Calculate correlation matrices ####
                user_scores = userscoreset[key]["user_scores"][...]
                machine_scores = generate_scores(probs, num_frames, positions)
                spear_avg_corr = evaluate_scores(machine_scores, user_scores, metric="spearmanr")
                kendal_avg_corr = evaluate_scores(machine_scores, user_scores, metric="kendalltau")

                spear_avg_corrs.append(spear_avg_corr)
                kendal_avg_corrs.append(kendal_avg_corr)

                if args.verbose:
                    table.append([key_idx+1, key, "{:.1%}".format(fm)])

                if args.save_results:
                    h5_res.create_dataset(key + '/user_summary', data=user_summary)
                    h5_res.create_dataset(key + '/score', data=probs)
                    h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                    h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                    h5_res.create_dataset(key + '/fm', data=fm)
        else:
            for key_idx, key in enumerate(test_keys):
                seq = dataset[key]['features'][...]
                seq = torch.from_numpy(seq).unsqueeze(0)
                if use_gpu: seq = seq.cuda()
                probs = model(seq)
                probs = probs.data.cpu().squeeze().numpy()


                cps = dataset[key]['change_points'][...]
                num_frames = dataset[key]['n_frames'][()]
                nfps = dataset[key]['n_frame_per_seg'][...].tolist()
                positions = dataset[key]['picks'][...]
                user_summary = dataset[key]['user_summary'][...]

                gtscore = dataset[key]['gtscore'][...]

                #print(key, len(probs), len(gtscore))

                machine_summary, _ = vsum_tools.generate_summary(probs, gtscore, cps, num_frames, nfps, positions)
                fm, _, _ = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric)
                fms.append(fm)

                #### Calculate correlation matrices ####
                user_scores = userscoreset[key]["user_scores"][...]
                machine_scores = generate_scores(probs, num_frames, positions)
                spear_avg_corr = evaluate_scores(machine_scores, user_scores, metric="spearmanr")
                kendal_avg_corr = evaluate_scores(machine_scores, user_scores, metric="kendalltau")

                spear_avg_corrs.append(spear_avg_corr)
                kendal_avg_corrs.append(kendal_avg_corr)

                if args.verbose:
                    table.append([key_idx+1, key, "{:.1%}".format(fm)])

                if args.save_results:
                    h5_res.create_dataset(key + '/user_summary', data=user_summary)
                    h5_res.create_dataset(key + '/score', data=probs)
                    h5_res.create_dataset(key + '/machine_summary', data=machine_summary)
                    h5_res.create_dataset(key + '/gtscore', data=dataset[key]['gtscore'][...])
                    h5_res.create_dataset(key + '/fm', data=fm)

    #if args.verbose:
    #    print(tabulate(table))
        
    if args.save_results: h5_res.close()

    mean_fm = np.mean(fms)
    print("Average F-score {:.3%}".format(mean_fm))

    mean_spear_avg = np.mean(spear_avg_corrs)
    mean_kendal_avg = np.mean(kendal_avg_corrs)
    print("Average Kendal {}".format(mean_kendal_avg))
    print("Average Spear {}".format(mean_spear_avg))

    return mean_fm

if __name__ == '__main__':
    main()

