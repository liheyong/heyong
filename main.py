"""
Created on Nov 10, 2017
Main function

@author: Lianhai Miao
"""
from model.agree import AGREE
import torch
import torch.optim as optim
import numpy as np
from time import time
from config import Config
from utils.util import Helper
from dataset import GDataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# 训练模型
def training(model, train_loader, epoch_id, config, type_m):
    # user trainning
    learning_rates = config.lr
    # 根据迭代的次数进行学习率的衰减
    lr = learning_rates[0]
    if epoch_id >= 15 and epoch_id < 25:
        lr = learning_rates[1]
    elif epoch_id >=20:
        lr = learning_rates[2]
    # lr decay
    if epoch_id % 5 == 0:
        lr /= 2

    # 定义优化器
    optimizer = optim.RMSprop(model.parameters(), lr)

    losses = []
    #开始训练
    for batch_id, (u, pi_ni) in enumerate(train_loader):
        # Data Load
        user_input = u.to(device)
        pos_item_input = pi_ni[:, 0].to(device)
        neg_item_input = pi_ni[:, 1].to(device)
        # Forward
        if type_m == 'user':
            pos_prediction = model(None, user_input, pos_item_input)
            neg_prediction = model(None, user_input, neg_item_input)
        elif type_m == 'group':
            pos_prediction = model(user_input, None, pos_item_input)
            neg_prediction = model(user_input, None, neg_item_input)
        # Zero_grad
        model.zero_grad()
        # Loss
        loss = torch.mean((pos_prediction - neg_prediction -1) **2)
        # record loss history
        losses.append(loss)  
        # Backward
        loss.backward()
        optimizer.step()
    print('Iteration %d, loss is [%.4f ]' % (epoch_id, torch.mean(torch.stack(losses), 0)))

def evaluation(model, helper, testRatings, testNegatives, K, type_m):
    model.eval()
    (hits, ndcgs) = helper.evaluate_model(model, testRatings, testNegatives, K, type_m)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    return hr, ndcg

if __name__ == '__main__':
    # 初始化参数
    config = Config()

    # 初始化工具函数
    helper = Helper()

    # get the dict of follow in user
    u_f_d = helper.gen_user_follow_dict(config.follow_in_user_path2)
    # 加载群组内的用户，构成dict
    # {groupid: [uid1, uid2, ..], ...} 组用户数据
    g_m_d = helper.gen_group_member_dict(config.user_in_group_path2)

    # 初始化数据类 在这里可以决定使用哪个数据
    dataset = GDataset(config.user_dataset2, config.group_dataset2, config.num_negatives)

    # 获取群组的数目、训练集中用户的数目、训练集中物品的数目
    num_group = len(g_m_d)
    num_users, num_items = dataset.num_users, dataset.num_items
    print("num_group is: " + str(num_group))
    print("num_users is: " + str(num_users))
    print("num_items is: " + str(num_items))

    # 训练 AGREE 模型
    agree = AGREE(num_users, num_items, num_group, config.num_follow,config.embedding_size, g_m_d, u_f_d,config.drop_ratio).to(device)

    # 打印配置信息
    print("AGREE 的Embedding 维度为: %d, 迭代次数为: %d, NDCG、HR评估选择topK: %d" %(config.embedding_size, config.epoch, config.topK))

    # 训练模型
    for epoch in range(config.epoch):
        agree.train()
        # 开始训练时间
        t1 = time()

        for _ in range(config.balance):
            training(agree, dataset.get_group_dataloader(config.batch_size), epoch, config, 'group')
        training(agree, dataset.get_user_dataloader(config.batch_size), epoch, config, 'user')

        print("user and group training time is: [%.1f s]" % (time()-t1))
        # 评估模型
        t2 = time()
        u_hr, u_ndcg = evaluation(agree, helper, dataset.user_testRatings, dataset.user_testNegatives, config.topK, 'user')
        print('User Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, [%.1f s]' % (
            epoch, time() - t1, u_hr, u_ndcg, time() - t2))

        hr, ndcg = evaluation(agree, helper, dataset.group_testRatings, dataset.group_testNegatives, config.topK, 'group')
        print(
            'Group Iteration %d [%.1f s]: HR = %.4f, '
            'NDCG = %.4f, [%.1f s]' % (epoch, time() - t1, hr, ndcg, time() - t2))

    print("Done!")










