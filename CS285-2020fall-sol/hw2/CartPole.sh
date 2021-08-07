source activate cs285
pip install -e .

python cs285/scripts/run_hw2.py --env_name CartPole-v0 -n 400 -b 1000 \
-dsa --exp_name q1_sb_no_rtg_dsa
