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
- 5-10fea_gen.ipynb 数据集生成,特征生成
- 5-11lgb.ipynb: 模型训练，评估
- gen_submit.ipynb: 模型融合，生成提交文件

## 4.运行流程
- 新建data目录，下载比赛数据集，放在data目录下
- 数据预处理：运行 处理问号和空值.ipynb
- 数据集生成：运行5-10fea_gen.ipynb
- 模型训练，评估：运行5-11lgb.ipynb
- 模型融合，生成提交文件： gen_submit.ipynb

## 5.模型及参数
模型：
-lgb
-xgb
-catboost

## 6.模型结果

| 线下auc | 线上auc | 
| ----------- | ------------ | 
| 0.8664    | 0.862     |
单模分数b榜在0.861-0.862左右
lgb_rank_res1 = pd.read_csv('./output/final_output/lgb_b_add_cols_final_score0.8657.txt',sep=' ',header=None)
lgb_rank_res2 = pd.read_csv('./output/final_output/lgb_final_usual_score8625.txt',sep=' ',header=None)
lgb_rank_res3 = pd.read_csv('./output/final_output/merge_seeds_lgb+xgb.txt',sep=' ',header=None)

b榜 86392
cat_rank_res1 = pd.read_csv('./output/seeds/cat_11_remain174fea_test_corr_cols_rank_score0.8653.txt',sep=' ',header=None)
cat_rank_res2 = pd.read_csv('./output/seeds/cat_47_remain174fea_test_corr_cols_rank_score0.8661.txt',sep=' ',header=None)
cat_rank_res3 = pd.read_csv('./output/seeds/cat_38_remain174fea_test_corr_cols_rank_score0.8652.txt',sep=' ',header=None)
