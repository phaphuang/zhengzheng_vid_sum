import torch
import sys

def compute_reward(seq, actions, ignore_far_sim=True, temp_dist_thre=20, use_gpu=False):
    """
    Compute diversity reward and representativeness reward

    Args:
        seq: sequence of features, shape (1, seq_len, dim)
        actions: binary action sequence, shape (1, seq_len, 1)
        ignore_far_sim (bool): whether to ignore temporally distant similarity (default: True)
        temp_dist_thre (int): threshold for ignoring temporally distant similarity (default: 20)
        use_gpu (bool): whether to use GPU
    """
    _seq = seq.detach()
    _actions = actions.detach()
    #pick_idxs = _actions.squeeze().nonzero().squeeze()
    pick_idxs = _actions.view(-1)
    pick_idxs = pick_idxs.nonzero()
    pick_idxs = pick_idxs.view(-1)
    #print('pick_idxs = ',pick_idxs.size())
    #pick_idxs = _actions.view(-1,1).nonzero().squeeze()
#    pick_idxs2 = pick_idxs1.nonzero()
#    pick_idxs3 = pick_idxs2.view(-1)
    if pick_idxs.numel()==0 or pick_idxs.size(0)==0 or pick_idxs.size(0)==None:
        print('one of the reward = 0')
        num_picks = 0
    else:
        #print('pick_idxs',pick_idxs.size(),end = ' ')
        num_picks = len(pick_idxs) if pick_idxs.ndimension() > 0 else 1
    
    if num_picks == 0:
        # give zero reward is no frames are selected
        reward = torch.tensor(0.)
        if use_gpu: reward = reward.cuda()
        return reward

    _seq = _seq.squeeze()
    #print('_seq = ',_seq.size())
    #print('_seq',_seq.size())
    n = _seq.size(0)

    # 计算多样性
    if num_picks == 1:
        reward_div = torch.tensor(0.)
        if use_gpu: reward_div = reward_div.cuda()
    else: 
        normed_seq = _seq / _seq.norm(p=2, dim=1, keepdim=True) #标准化，并平方
        dissim_mat = 1. - torch.matmul(normed_seq, normed_seq.t()) # 不相似矩阵，matmul矩阵乘法 .t为转置矩阵 dissimilarity matrix [Eq.4]
        dissim_submat = dissim_mat[pick_idxs,:][:,pick_idxs] #选择所选帧的不相似矩阵
        if ignore_far_sim:
            # 距离高于某个阈值时，定义不相似值为1
            pick_mat = pick_idxs.expand(num_picks, num_picks)
            temp_dist_mat = torch.abs(pick_mat - pick_mat.t())
            dissim_submat[temp_dist_mat > temp_dist_thre] = 1.
        reward_div = dissim_submat.sum() / (num_picks * (num_picks - 1.)) # diversity reward [Eq.3]

    # 计算 representativeness reward
    dist_mat = torch.pow(_seq, 2).sum(dim=1, keepdim=True).expand(n, n) #扩展成n*n矩阵
    dist_mat = dist_mat + dist_mat.t()
    dist_mat.addmm_(1, -2, _seq, _seq.t())
    dist_mat = dist_mat[:,pick_idxs]
    dist_mat = dist_mat.min(1, keepdim=True)[0]
    reward_rep = torch.exp(-dist_mat.mean())

    # 结合两种
    reward = (reward_div + reward_rep) * 0.5

    return reward
