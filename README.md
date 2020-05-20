# Attentive-Group-Recommendation

## 说明

该项目Fork LianHaiMiao/Attentive-Group-Recommendation

原代码地址：https://github.com/LianHaiMiao/Attentive-Group-Recommendation

之所以fork这个库是因为，源库中的代码是基于以下版本实现的
- pytorch version:  '0.3.0'
- python version: '3.5'

pytorch目前已经更新到 1.4.0版本，所以进行代码的更新，同时将Python版本升级到3.7（python 版本影响不大）


```
@inproceedings{Cao2018Attentive,
 author = {Cao, Da and He, Xiangnan and Miao, Lianhai and An, Yahui and Yang, Chao and Hong, Richang},
 title = {Attentive Group Recommendation},
 booktitle = {The 41st International ACM SIGIR Conference on Research \&\#38; Development in Information Retrieval},
 series = {SIGIR '18},
 year = {2018},
 isbn = {978-1-4503-5657-2},
 location = {Ann Arbor, MI, USA},
 pages = {645--654},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/3209978.3209998},
 doi = {10.1145/3209978.3209998},
 acmid = {3209998},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {atention mechanism, cold-start problem, group recommendation, neural collaborative filtering, recommender systems},
}
```

## 我的环境
- pytorch version: '1.4.0'
- python version: '3.7'

## 代码运行

Run AGREE:

```
python main.py
```

训练之后，数据集进行测试输出为：

```
AGREE at embedding size 32, run Iteration:30, NDCG and HR at 5
...
User Iteration 10 [449.8 s]: HR = 0.6216, NDCG = 0.4133, [1.0 s]
Group Iteration 10 [471.9 s]: HR = 0.5910, NDCG = 0.4005, [23.0 s]

```


## 参数说明

将模型使用的参数放置在 utils/config.py 中

## 数据集

这里只开源了CAMRa2011数据集，因为马蜂窝的数据集被使用在另一篇算法中，因为算法还未公开，所以马蜂窝的数据集暂时不公开

data/CAMRa2011 数据说明：
- groupMember.txt
  - 每一行包含的数据格式为：群组id 用户id1,用户id2,..
- groupRatingNegative.txt
    - 每一行包含的数据格式为：(群组id,物品id) 负采样的物品id1,负采样的物品id2,...
    - 每一行的负样本包含100个
    - 负采样的物品ID表示群组内用户没有交互行为的物品
    
- groupRatingTest.txt
    - 每一行有三个数据分别表示：群组ID，物品ID，评分
- groupRatingTrain.txt
    - 每一行有三个数据分别表示：群组ID，物品ID，评分
- userRatingNegative.txt
    - 每一行包含的数据格式为：(群组id,物品id) 负采样的物品id1,负采样的物品id2,...
    - 每一行的负样本包含100个
    - 负采样的物品ID表示用户没有交互行为的物品
- userRatingTest.txt
    - 每一行有三个数据分别表示：用户ID，物品ID，评分
- userRatingTrain.txt
    - 每一行有三个数据分别表示：用户ID，物品ID，评分