source activate cs285

python cs285/scripts/run_hw1.py \
--expert_policy_file cs285/policies/experts/Ant.pkl \
--env_name Ant-v2 --exp_name dagger_ant --n_iter 10 \
--do_dagger --expert_data cs285/expert_data/expert_data_Ant-v2.pkl \
--ep_len 1000 \
--eval_batch_size 10000 \
--video_log_freq -1

