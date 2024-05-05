CUDA_VISIBLE_DEVICES='0' python main.py --rounds 2 --id 1 &> logs/log_1.log
CUDA_VISIBLE_DEVICES='0' python main.py --rounds 3 --id 2 &> logs/log_2.log
CUDA_VISIBLE_DEVICES='0' python main.py --rounds 5 --id 3 &> logs/log_3.log
CUDA_VISIBLE_DEVICES='0' python main.py --rounds 10 --id 4 &> logs/log_4.log
CUDA_VISIBLE_DEVICES='0' python main.py --n_clients 2 --id 5 &> logs/log_5.log
CUDA_VISIBLE_DEVICES='0' python main.py --n_clients 4 --id 6 &> logs/log_6.log
CUDA_VISIBLE_DEVICES='0' python main.py --n_clients 6 --id 7 &> logs/log_7.log
CUDA_VISIBLE_DEVICES='0' python main.py --n_clients 8 --id 8 &> logs/log_8.log