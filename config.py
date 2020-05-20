'''
Created on Nov 10, 2017
Store parameters

@author: Lianhai Miao
'''

class Config(object):
    def __init__(self):
        self.path1 = './data/CAMRa2011/'
        self.user_dataset1 = self.path1 + 'userRating'
        self.group_dataset1 = self.path1 + 'groupRating'
        self.user_in_group_path1 = "./data/CAMRa2011/groupMember.txt"


        self.path2 = './data/MaFengWo/'
        self.user_dataset2 = self.path2 + 'userRating'
        self.group_dataset2 = self.path2 + 'groupRating'
        self.user_in_group_path2 = "./data/MaFengWo/groupMember.txt"
        self.follow_in_user_path2 = "./data/MaFengWo/userFollow.txt"

        self.embedding_size = 32
        self.epoch = 30        #30 改
        self.num_negatives = 6
        self.batch_size = 256
        self.lr = [0.000005, 0.000001, 0.0000005]
        self.drop_ratio = 0.2
        self.topK = 5
        self.num_follow = 13096
        self.balance = 6
