'''
Created on Nov 10, 2017
Create Model

@author: Lianhai Miao
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class AGREE(nn.Module):
    def __init__(self, num_users, num_items, num_groups, num_follow,embedding_dim, group_member_dict,user_follow_dict,drop_ratio):
        super(AGREE, self).__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.groupembeds = GroupEmbeddingLayer(num_groups, embedding_dim)
        self.followembeds = FollowEmebddingLayer(num_follow, embedding_dim)
        # ??
        self.followAttention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.predictlayer = PredictLayer(3 * embedding_dim, drop_ratio)
        self.group_member_dict = group_member_dict
        self.user_follow_dict = user_follow_dict


        self.num_follow = num_follow
        self.num_users = num_users
        self.num_groups = len(self.group_member_dict)
        # initial model 初始化模型，收敛

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward(self, group_inputs, user_inputs, item_inputs):
        # train group
        if (group_inputs is not None) and (user_inputs is None):
            out = self.grp_forward(group_inputs, item_inputs)
        # train user
        else:
            out = self.usr_forward(user_inputs, item_inputs)
        return out

    # group forward
    def grp_forward(self, group_inputs, item_inputs):
        group_embeds = Variable(torch.Tensor())
        item_embeds_full = self.itemembeds(Variable(torch.LongTensor(item_inputs)))
        for i, j in zip(group_inputs, item_inputs):
            members = self.group_member_dict[i.item()]
            members_embeds = self.userembeds(Variable(torch.LongTensor(members)))   #获得组成员嵌入
            items_numb = []
            for _ in members:
                items_numb.append(j)
            item_embeds = self.itemembeds(Variable(torch.LongTensor(items_numb)))
            group_item_embeds = torch.cat((members_embeds, item_embeds), dim=1)

            at_wt = self.attention(group_item_embeds)      #获得组和项目之间的 注意力权重。改进这一部分权重

            g_embeds_with_attention = torch.matmul(at_wt, members_embeds)
            group_embeds_pure = self.groupembeds(Variable(torch.LongTensor([i])))
            g_embeds = g_embeds_with_attention + group_embeds_pure
            if group_embeds.dim() == 0:
                group_embeds = g_embeds
            else:
                group_embeds = torch.cat((group_embeds, g_embeds))

        element_embeds = torch.mul(group_embeds, item_embeds_full)  # Element-wise product

        new_embeds = torch.cat((element_embeds, group_embeds, item_embeds_full), dim=1)

        y = torch.sigmoid(self.
                          (new_embeds))  #Shared Hidden layers.

        #print(y.detach().numpy().shape)
        return y    #128 1

    # user follow aggregate
    def user_aggregate(self, user_inputs):
        user_finnal_list = []
        for i in user_inputs:
            follows = self.user_follow_dict[i]
            follow_embeds = self.followembeds(Variable(torch.LongTensor(follows)))
            users_numb = len(follows)
            # user embedding
            user_embeds = self.userembeds(Variable(torch.LongTensor( [i]*users_numb )))
            user_follow_embeds = torch.cat((follow_embeds, user_embeds), dim=1)
            at_wt = self.followAttention(user_follow_embeds)

            u_embeds_with_attention = torch.matmul(at_wt, follow_embeds)

            user_embeds_pure = self.userembeds(Variable(torch.LongTensor([i])))
            u_embeds = u_embeds_with_attention + user_embeds_pure
            user_finnal_list.append(u_embeds.view(-1))
        user_finnal_vec = torch.stack(user_finnal_list, dim=0)
        return user_finnal_vec

    # user forward 用户-项目交互
    def usr_forward(self, user_inputs, item_inputs):
        user_inputs_var, item_inputs_var = Variable(user_inputs), Variable(item_inputs) #构造变量，数据流

        user_embeds = self.userembeds(user_inputs_var)
        item_embeds = self.itemembeds(item_inputs_var)

        element_embeds = torch.mul(user_embeds, item_embeds)  # Element-wise product

        new_embeds = torch.cat((element_embeds, user_embeds, item_embeds), dim=1)     #128*96
        #print(new_embeds.detach().numpy().shape)


        y = torch.sigmoid(self.predictlayer(new_embeds)) #128 1
        #print(y.detach().numpy().shape)
        return y


def tensor2np(self, tens):
    return tens.numpy()

class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds

class FollowEmebddingLayer(nn.Module):
    def __init__(self, num_follow, embedding_dim):
        super(FollowEmebddingLayer, self).__init__()
        self.followEmbedding = nn.Embedding(num_follow, embedding_dim)

    def forward(self, follow_inputs):
        follow_embeds = self.followEmbedding(follow_inputs)
        return follow_embeds

class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class GroupEmbeddingLayer(nn.Module):
    def __init__(self, number_group, embedding_dim):
        super(GroupEmbeddingLayer, self).__init__()
        self.groupEmbedding = nn.Embedding(number_group, embedding_dim)

    def forward(self, num_group):
        group_embeds = self.groupEmbedding(num_group)
        return group_embeds


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__() #继承初始化？ 什么意思？
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        out = self.linear(x)
        weight = F.softmax(out.view(1, -1), dim=1)
        return weight


class PredictLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(PredictLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 8),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        return out