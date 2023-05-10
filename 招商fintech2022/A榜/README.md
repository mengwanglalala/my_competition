# 招商银行Fintech挑战赛

## 1.环境配置
- python3
- numpy
- pandas 
- scikit-learn 
- lightgbm
- xgboost
- catboost

## 2.运行配置
- CPU/GPU均可


## 3.目录结构
- fea_gen.ipynb 数据集生成,特征生成
- lgb2.ipynb: 模型训练，评估
- gen_submit.ipynb: 模型融合，生成提交文件

## 4.运行流程
- 新建data目录，下载比赛数据集，放在data目录下
- 预处理：运行 处理问号和空值.ipynb
- 数据集生成：运行fea_gen.ipynb
- 模型训练，评估：运行lgb2.ipynb

## 5.模型及参数
模型：
-lgb
-xgb
-catboost

## 6.模型结果

| 线下auc | 线上auc | 
| ----------- | ------------ | 
| 0.9509    | 0.9522     |
最终提交文件位
merge_rank_lgb0.9510xgb0.9509.txt
