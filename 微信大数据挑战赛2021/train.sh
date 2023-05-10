cd src/prepare/
/opt/conda/envs/tensorflow_py3/bin/python test.py
/opt/conda/envs/tensorflow_py3/bin/python P0_conver_csv_to_feather.py
/opt/conda/envs/tensorflow_py3/bin/python F0_statis_5day_feature.py
/opt/conda/envs/tensorflow_py3/bin/python F1_feed_embedding_process.py
/opt/conda/envs/tensorflow_py3/bin/python F5_user_author_n2v.py
/opt/conda/envs/tensorflow_py3/bin/python F6_deepwalk.py

cd ..
cd train
/opt/conda/envs/tensorflow_py3/bin/python test.py
/opt/conda/envs/tensorflow_py3/bin/python P0_generate_training_data.py

/opt/conda/envs/tensorflow_py3/bin/python T0_seed.py 2021
/opt/conda/envs/tensorflow_py3/bin/python T0_seed.py 1998
/opt/conda/envs/tensorflow_py3/bin/python T0_seed.py 4096
/opt/conda/envs/tensorflow_py3/bin/python T0_seed.py 123
/opt/conda/envs/tensorflow_py3/bin/python T0_seed.py 47
/opt/conda/envs/tensorflow_py3/bin/python T0_seed.py 520
/opt/conda/envs/tensorflow_py3/bin/python T0_seed.py 6666
/opt/conda/envs/tensorflow_py3/bin/python T0_seed.py 1010
/opt/conda/envs/tensorflow_py3/bin/python T1_0_readcom_like_comment_click.py 182
/opt/conda/envs/tensorflow_py3/bin/python T1_0_readcom_like_comment_click.py 2021
/opt/conda/envs/tensorflow_py3/bin/python T1_0_readcom_like_comment_click.py 22

/opt/conda/envs/tensorflow_py3/bin/python T1_1_like_favor_forward.py 182
/opt/conda/envs/tensorflow_py3/bin/python T1_1_like_favor_forward.py 2021
/opt/conda/envs/tensorflow_py3/bin/python T1_1_like_favor_forward.py 22

/opt/conda/envs/tensorflow_py3/bin/python T1_2_click_follow.py 182
/opt/conda/envs/tensorflow_py3/bin/python T1_2_click_follow.py 2021
/opt/conda/envs/tensorflow_py3/bin/python T1_2_click_follow.py 22

/opt/conda/envs/tensorflow_py3/bin/python AutoInt_more_seed.py
/opt/conda/envs/tensorflow_py3/bin/python xdeepfm_more_seed.py

cd ..
cd ..
