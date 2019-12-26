#python -u ga_attack.py --n_fake_users=1 --pop_size=100 --max_pop_size=120 --train_frac=0.5 > n_users_1.txt &
# tensorboard --logdir /home/lidora/stuff/GA_Attack/runs --port 4445 --host deml-prod-worker-03.va2
# sets priority of 10, where default is zero, non-urgent will have positivetmux number, which means, these processes are nice!

#python -u ga_attack_train_baseline.py --n_fake_users=2 &
#python -u ga_attack_train_baseline.py --n_fake_users=4  &
#python -u ga_attack_train_baseline.py --n_fake_users=8  &
#python -u ga_attack_train_baseline.py --n_fake_users=16  &
#python -u ga_attack_train_baseline.py --n_fake_users=32  &
#python -u ga_attack_train_baseline.py --n_fake_users=64  &
#python -u ga_attack_train_baseline.py --n_fake_users=128 &
python -u ga_attack_train_baseline.py --n_fake_users=256 &
python -u ga_attack_train_baseline.py --n_fake_users=512 &
python -u ga_attack_train_baseline.py --n_fake_users=1024 &

wait
echo 'finished training all models'