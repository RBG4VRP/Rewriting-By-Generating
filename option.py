import os
import time
import argparse
import torch

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',default=100, type=int, help='random seed.')
    parser.add_argument('--epoch_size',default=100, type=int, help='')
    parser.add_argument('--num_epochs',default=50, type=int, help='number of epochs')
    parser.add_argument('--batch_size',default=10, type=int, help='training batch size')
    parser.add_argument('--eval_batch_size',default=20, type=int, help='evaluation batch size')
    parser.add_argument('--val_size',default=20, type=int, help='size of validation set at training')
    parser.add_argument('--near_K',default=-1, type=int, help='')
    parser.add_argument('--save_epoch',default=5, type=int, help='epoch for saving model')
    parser.add_argument('--K_means_step',default=10, type=int, help='')
    parser.add_argument('--iter_step_train',default=10, type=int, help='')
    parser.add_argument('--iter_step_eval',default=100, type=int, help='')
    parser.add_argument('--size',default=500, type=int, help='CVRP problem size.')
    parser.add_argument('--lr1',default=3e-5, type=float, help='learning rate 1')
    parser.add_argument('--lr2',default=1e-3, type=float, help='learning rate 2')
    parser.add_argument('--beta',default=0.1, type=float, help='beta.')
    parser.add_argument('--always_update',default=False,action='store_true')
    parser.add_argument('--proctitle',default='train',type=str)
    parser.add_argument('--save_model',default=None,type=str)
    parser.add_argument('--load_model', default='saved/model500_99.pt', type=str)
    parser.add_argument('--eval_data', default=None, type=str)
    parser.add_argument('--eval_scale_factor', default=1, type=float)
    # parser.add_argument('--model_id',default=0,type=int)  #0:randomly choose action each time.
    parser.add_argument('--update_model_step',default=12,type=int)
    parser.add_argument('--model_type',default=1,type=int)
    parser.add_argument('--enable_random_rotate_train',action='store_true',default=True)
    parser.add_argument('--enable_random_rotate_eval',action='store_true',default=True)
    parser.add_argument('--up_rate_train', default=0.005, type=float)
    parser.add_argument('--up_rate_eval', default=0.0005, type=float)
    parser.add_argument('--rl_steps', default=1, type=int)
    parser.add_argument('--enable_gradient_clipping', action='store_true', default= True)
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--random_selection', default=False, action='store_true')
    parser.add_argument('--train_selection_only', default=False, action='store_true')
    parser.add_argument('--train_from_scratch', default=False, action='store_true')
    parser.add_argument('--train_at_train', default=False, action='store_true')
    parser.add_argument('--fix_select_model', default=True, action='store_true')
    parser.add_argument('--enable_running_cost', default=True, action='store_true')
    parser.add_argument('--running_cost_alpha', default=0.99, type=float)

    args= parser.parse_args()
    if args.near_K==-1:
        if args.size in [500,1000,2000]:
            args.near_K={500:5,1000:8,2000:9}[args.size]
        else:
            args.near_K=None
    if args.save_model is None:
        args.save_model='model{}'.format(args.size)
    return args