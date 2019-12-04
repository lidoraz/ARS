#python -u ga_attack.py --n_fake_users=1 --pop_size=100 --max_pop_size=120 --train_frac=0.5 > n_users_1.txt &
# tensorboard --logdir /home/lidora/stuff/GA_Attack/runs --port 4445 --host deml-prod-worker-03.va2
# sets priority of 10, where default is zero, non-urgent will have positivetmux number, which means, these processes are nice!
nice -10 python -u ga_attack.py --n_fake_users=2 --pop_size=100 --max_pop_size=300 --train_frac=1.0 > ../logs/n_users_2.txt  &
nice -10 python -u ga_attack.py --n_fake_users=4 --pop_size=100 --max_pop_size=300 --train_frac=1.0 > ../logs/n_users_4.txt  &
nice -10 python -u ga_attack.py --n_fake_users=8 --pop_size=100 --max_pop_size=300 --train_frac=1.0 > ../logs/n_users_8.txt &
nice -10 python -u ga_attack.py --n_fake_users=12 --pop_size=100 --max_pop_size=300 --train_frac=1.0 > ../logs/n_users_12.txt &
nice -10 python -u ga_attack.py --n_fake_users=16 --pop_size=100 --max_pop_size=300 --train_frac=1.0 > ../logs/n_users_16.txt &
nice -10 python -u ga_attack.py --n_fake_users=32 --pop_size=100 --max_pop_size=300 --train_frac=1.0 > ../logs/n_users_32.txt &
nice -10 python -u ga_attack.py --n_fake_users=64 --pop_size=100 --max_pop_size=300 --train_frac=1.0 > ../logs/n_users_64.txt &
nice -10 python -u ga_attack.py --n_fake_users=128 --pop_size=100 --max_pop_size=300 --train_frac=1.0 > ../logs/n_users_128.txt  &
# python -u ga_attack.py --n_fake_users=10 --pop_size=100 --max_pop_size=120 --train_frac=0.5 &;


