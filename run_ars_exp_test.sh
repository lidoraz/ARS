##python -u ga_attack.py --n_fake_users=1 --pop_size=100 --max_pop_size=120 --train_frac=0.5 > n_users_1.txt &
## tensorboard --logdir /home/lidora/stuff/GA_Attack/runs --port 4445 --host deml-prod-worker-03.va2
## sets priority of 10, where default is zero, non-urgent will have positivetmux number, which means, these processes are nice!
#
#
##nohup nice -10 python ga_attack_multiprocess.py --n_processes=8 --n_fake_users=2 --selection='TOURNAMENT'  --pop_size=700 --max_pop_size=0 --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=8 --n_fake_users=4 --selection='TOURNAMENT'  --pop_size=700 --max_pop_size=0 --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=8 --n_fake_users=8 --selection='TOURNAMENT'  --pop_size=700 --max_pop_size=0 --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=8 --n_fake_users=16 --selection='TOURNAMENT' --pop_size=700 --max_pop_size=0 --n_generations=100 --train_frac=0.01 &
##nohup nice -10 python ga_attack_multiprocess.py --n_processes=8 --n_fake_users=32 --selection='TOURNAMENT'  --pop_size=700 --max_pop_size=0 --n_generations=100 --train_frac=0.01 &
#
##nohup nice -10 python ga_attack_multiprocess.py --n_processes=8 --n_fake_users=2  --selection='ROULETTE'  --pop_size=1000 --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=8 --n_fake_users=4  --selection='ROULETTE'  --pop_size=2000 --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=8 --n_fake_users=8  --selection='ROULETTE'  --pop_size=2000 --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=8 --n_fake_users=16 --selection='ROULETTE' --pop_size=2000  --n_generations=100 --train_frac=0.01 &
##nohup nice -10 python ga_attack_multiprocess.py --n_processes=8 --n_fake_users=32  --selection='ROULETTE'  --pop_size=1000 --n_generations=100 --train_frac=0.01 &
#
#
## עדיין צריך להריץ את הריצות האלה הרנדומליות, ואז ברגע שמסיימים איתן, ניתן יהיה להריץ את כל התהליף הזה עם כל היוזרים.
##nohup nice -15 python ga_attack_multiprocess.py --n_processes=4 --n_fake_users=2  --selection='RANDOM'  --pop_size=1000 --n_generations=100  --train_frac=0.01 &
#nohup nice -15 python ga_attack_multiprocess.py --n_processes=16 --n_fake_users=4  --selection='RANDOM'  --pop_size=1000 --n_generations=100  --train_frac=0.01 &
#nohup nice -15 python ga_attack_multiprocess.py --n_processes=16 --n_fake_users=8  --selection='RANDOM'  --pop_size=1000 --n_generations=100  --train_frac=0.01 &
#nohup nice -15 python ga_attack_multiprocess.py --n_processes=16  --n_fake_users=16 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 &
##nohup nice -15 python ga_attack_multiprocess.py --n_processes=4 --n_fake_users=32  --selection='RANDOM'  --pop_size=1000 --n_generations=100  --train_frac=0.01 &
#
#
#
#nohup nice -15 python ga_attack_multiprocess.py --n_processes=4  --n_fake_users=256 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=4 --n_fake_users=256 --selection='ROULETTE'  --pop_size=2000 --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=4 --n_fake_users=256  --selection='TOURNAMENT'  --pop_size=700 --max_pop_size=0 --n_generations=100 --train_frac=0.01 &
#nohup nice -15 python ga_attack_multiprocess.py --n_processes=4  --n_fake_users=512 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=4 --n_fake_users=512 --selection='ROULETTE'  --pop_size=2000 --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=4 --n_fake_users=512  --selection='TOURNAMENT'  --pop_size=700 --max_pop_size=0 --n_generations=100 --train_frac=0.01 &
#nohup nice -15 python ga_attack_multiprocess.py --n_processes=4  --n_fake_users=1024 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=4 --n_fake_users=1024 --selection='ROULETTE'  --pop_size=2000 --n_generations=100 --train_frac=0.01 &
#nohup nice -10 python ga_attack_multiprocess.py --n_processes=4 --n_fake_users=1024  --selection='TOURNAMENT'  --pop_size=700 --max_pop_size=0 --n_generations=100 --train_frac=0.01 &


python test_ga_attack_multiprocess.py --n_processes=6  --n_fake_users=2 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 --save=False &
python test_ga_attack_multiprocess.py --n_processes=6  --n_fake_users=4 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 --save=False &
python test_ga_attack_multiprocess.py --n_processes=6  --n_fake_users=8 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 --save=False &
python test_ga_attack_multiprocess.py --n_processes=6  --n_fake_users=16 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 --save=False &
python test_ga_attack_multiprocess.py --n_processes=6  --n_fake_users=32 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 --save=False &
python test_ga_attack_multiprocess.py --n_processes=6  --n_fake_users=64 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 --save=False &
python test_ga_attack_multiprocess.py --n_processes=6  --n_fake_users=128 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 --save=False &
python test_ga_attack_multiprocess.py --n_processes=6  --n_fake_users=256 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 --save=False &
python test_ga_attack_multiprocess.py --n_processes=4  --n_fake_users=512 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 --save=False &
python test_ga_attack_multiprocess.py --n_processes=4  --n_fake_users=1024 --selection='RANDOM' --pop_size=1000  --n_generations=100 --train_frac=0.01 --save=False &
