# **2021中国高校计算机大赛-微信大数据挑战赛**

本次比赛基于脱敏和采样后的数据信息，对于给定的一定数量到访过微信视频号“热门推荐”的用户，根据这些用户在视频号内的历史n天的行为数据，通过算法在测试集上预测出这些用户对于不同视频内容的互动行为（包括点赞、点击头像、收藏、转发等）的发生概率。 


## **1. 环境依赖**
- tensorflow>=1.12.1 / 1.15
- pandas>=1.0.5
- numpy>=1.16.4
- numba>=0.45.1
- scipy>=1.3.1
- scikit-learn>=0.24.2
- keras>=2.2.4
- pyarrow>=4.0.1
- deepctr>=0.8.6
- tqdm>=4.60.0
- gensim>=3.8.3
- node2vec>=0.4.3
- deepwalk>=1.0.3 
- jsonschema>=3.2.0
- python3

## **2. 目录结构**

```
./
├── README.md
├── requirements.txt, python package requirements 
├── init.sh, script for installing package requirements
├── train.sh, script for preparing train/inference data and training models, including pretrained models
├── inference.sh, script for inference 
├── src
│   ├── prepare, codes for preparing train/inference dataset
|       ├──v0_statis_featrue.py
|       ├──v1_feed_embeding_process.py 
|       ├──v5_author_user_n2vfeature.py
|       ├──0_prepare_training_data.py
│   ├── model, codes for model architecture
|       ├──mmoe.py  
|       ├──model.py 
|       ├──old_mmoe.py 
|   ├── train, codes for training
|       ├──0_all_data_train_base.py
|       ├──click_follow.py
|       ├──like_favor_forward.py
|       ├──readcom_like_comment_click.py
|   ├── inference.py, main function for inference on test dataset
|   ├── evaluation.py, main function for evaluation 
├── data
│   ├── wedata, dataset of the competition
│       ├── wechat_algo_data1, preliminary dataset
│   ├── submission, prediction result after running inference.sh
│   ├── fea_data, feature data
│   ├── feather_data, dataset of the competition
│   ├── train_data, train data
│   ├── cahce, train data
│   ├── model, model files (e.g. tensorflow checkpoints)
├── config, configuration files for your method (e.g. yaml file)
|       ├──conf.py 
```

## **3. 运行流程**
- 进入目录：cd /home/tione/notebook/wbdc2021-semi
- 安装环境：sh init.sh
- 数据准备和模型训练：sh train.sh
- 预测并生成结果文件：sh inference.sh ../wbdc2021/data/wedata/wechat_algo_data2/test_b.csv

## **4. 模型及特征**
- 模型：MMOE
- 参数：
    - batch_size: 4096
    - emded_dim: 32
    - num_epochs: 4
    - learning_rate: 0.04
- 特征：
    - dnn 特征: userid, feedid, authorid, bgm_singer_id, bgm_song_id，等
    - linear 特征：videoplayseconds, device，用户/feed 历史行为次数，等
    - 图特征：用户历史观看视频，用户历史观看视频作者，等

    
## **5. 算法性能**
- 资源配置：2*P40_48G显存_14核CPU_112G内存
- 预测耗时  
    - 总预测时长: 1291 s
    - 单个目标行为2000条样本的平均预测时长: 86.7471 ms


## **6. 代码说明**
模型预测部分代码位置如下：

| 路径 | 行数 | 内容 |
| :--- | :--- | :--- |
| src/inference.py | 305 | `pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 4)`|
| src/inference.py | 417 | `pred_ans = train_model.predict(test_model_input, batch_size=batch_size * 4)`|
| src/inference.py | 448 | `pred_split0_ans = train_model_split0.predict(test_model_input, batch_size=batch_size * 4)`|
| src/inference.py | 471 | `pred_split0_ans = train_model_split0.predict(test_model_input, batch_size=batch_size * 4)`|
| src/inference.py | 493 | `pred_split0_ans = train_model_split0.predict(test_model_input, batch_size=batch_size * 4)`|




   



