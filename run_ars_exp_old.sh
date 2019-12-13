##python -u ga_attack.py --n_fake_users=1 --pop_size=100 --max_pop_size=120 --train_frac=0.5 > n_users_1.txt &
## tensorboard --logdir /home/lidora/stuff/GA_Attack/runs --port 4445 --host deml-prod-worker-03.va2
## sets priority of 10, where default is zero, non-urgent will have positivetmux number, which means, these processes are nice!
#
#./run_ars_exp.sh
#wait
#echo 'finished training all models'
#
##nohup nice -10 python -u ga_attack.py --n_fake_users=2   --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u ga_attack.py --n_fake_users=4   --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u ga_attack.py --n_fake_users=8   --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u ga_attack.py --n_fake_users=12  --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u ga_attack.py --n_fake_users=16  --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u ga_attack.py --n_fake_users=32  --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u ga_attack.py --n_fake_users=64  --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01   &
##nohup nice -10 python -u ga_attack.py --n_fake_users=128 --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01   &
#
#nohup python -u ga_attack_multiprocess.py --mode=ATTACK --n_processes=12 --n_fake_users=2 --selection_mode='ROULETTE' --pop_size=2000 --train_frac=0.01 &
#nohup python -u ga_attack_multiprocess.py --mode=ATTACK --n_processes=12 --n_fake_users=8 --selection_mode='ROULETTE' --pop_size=2000 --train_frac=0.01 &
#nohup python -u ga_attack_multiprocess.py --mode=ATTACK --n_processes=12 --n_fake_users=16 --selection_mode='ROULETTE' --pop_size=2000 --train_frac=0.01 &
#nohup python -u ga_attack_multiprocess.py --mode=ATTACK --n_processes=12 --n_fake_users=32 --selection_mode='ROULETTE' --pop_size=2000 --train_frac=0.01 &
#
#nohup nice -10 python -u random_attack.py --n_fake_users=2   --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u random_attack.py --n_fake_users=4   --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
#nohup nice -10 python -u random_attack.py --n_fake_users=8   --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u random_attack.py --n_fake_users=12  --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
#nohup nice -10 python -u random_attack.py --n_fake_users=16  --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u random_attack.py --n_fake_users=32  --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u random_attack.py --n_fake_users=64  --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
##nohup nice -10 python -u random_attack.py --n_fake_users=128 --selection_mode='ROULETTE' --max_pop_size=2000 --train_frac=0.01  &
#
#
#
##                      THIS IS FOR ROULETTE!!!!!!! NOTE
#
#
#
##nohup nice -10 python -u ga_attack.py --n_fake_users=2   --pop_size=1000 --max_pop_size=0 --train_frac=0.00 --save_dir='/data/lidor_data/ars/agents' &
##nohup nice -10 python -u ga_attack.py --n_fake_users=4   --pop_size=1000 --max_pop_size=0 --train_frac=0.00 --save_dir='/data/lidor_data/ars/agents' &
##nohup nice -10 python -u ga_attack.py --n_fake_users=8   --pop_size=1000 --max_pop_size=0 --train_frac=0.00 --save_dir='/data/lidor_data/ars/agents' &
##nohup nice -10 python -u ga_attack.py --n_fake_users=12  --pop_size=1000 --max_pop_size=0 --train_frac=0.00 --save_dir='/data/lidor_data/ars/agents' &
##nohup nice -10 python -u ga_attack.py --n_fake_users=16  --pop_size=1000 --max_pop_size=0 --train_frac=0.00 --save_dir='/data/lidor_data/ars/agents' &
##nohup nice -10 python -u ga_attack.py --n_fake_users=32  --pop_size=1000 --max_pop_size=0 --train_frac=0.00 --save_dir='/data/lidor_data/ars/agents' &
##nohup nice -10 python -u ga_attack.py --n_fake_users=64  --pop_size=1000 --max_pop_size=0 --train_frac=0.00 --save_dir='/data/lidor_data/ars/agents' &
##nohup nice -10 python -u ga_attack.py --n_fake_users=128 --pop_size=1000 --max_pop_size=0 --train_frac=0.00 --save_dir='/data/lidor_data/ars/agents' &
#
## python -u ga_attack.py --n_fake_users=10 --pop_size=100 --max_pop_size=120 --train_frac=0.5