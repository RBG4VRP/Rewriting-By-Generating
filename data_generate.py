import torch
import numpy as np
import pickle
import argparse
import os
# import random

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',default=100, type=int, help='random seed, should be different from training and validation data')
    parser.add_argument('--num_samples',default=100, type=int, help='')
    parser.add_argument('--size',default=100, type=int, help='')
    parser.add_argument('--capacity',default=50, type=int, help='')
    parser.add_argument('--save_path',default='data', type=str, help='')

    parser.add_argument('--random_depot',action='store_true',default=False)
    
    args= parser.parse_args()
    return args



def generate_data(num_samples=10000,size=500,capacity=50,random_depot=False):
    # CAPACITIES=50

    if not random_depot:
        return [{'depot':torch.tensor([0.5,0.5]),'loc':torch.FloatTensor(size, 2).uniform_(0, 1),
             'demand':(torch.FloatTensor(size).uniform_(0, 9).int() + 1).float() / capacity} for i in range(num_samples)]



if __name__ == '__main__':
    args=parse()

    torch.manual_seed(args.seed)

    dataset=generate_data(args.num_samples, args.size, args.capacity, args.random_depot)

    filename='data_{}_{}_{}.pkl'.format(args.size,args.num_samples,args.capacity)

    with open(os.path.join(args.save_path,filename),'wb') as f:
            pickle.dump(dataset,f)

    print('dataset generated at {}.'.format(os.path.join(args.save_path,filename)))
