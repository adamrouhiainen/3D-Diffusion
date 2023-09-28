import argparse
from os import listdir, path
import torch
import numpy as np
from shutil import copy

def get_latest_experiment(experiments_dir, filter_str):
    experiments = sorted([e for e in listdir(experiments_dir) if path.isdir(f'{experiments_dir}/{e}')])
    experiments = list(filter(lambda e: filter_str in e, experiments))
    latest = experiments[-1]
    return latest

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--experiments-dir', type=str, default=f'experiments', help='Experiments directory')
    parser.add_argument('-d', '--dest', type=str, help='Image destination')
    parser.add_argument('-f', '--filter', type=str, default='', help='String to filter experiments by')

    args = parser.parse_args()

    latest = get_latest_experiment(args.experiments_dir, args.filter)

    latest_results_path = f'{args.experiments_dir}/{latest}/results/test/0'

    latest_outputs = [f for f in listdir(latest_results_path) if path.isfile(f'{latest_results_path}/{f}') and 'Out_' in f and '.pt' in f]
    
    assert len(latest_outputs) == 1

    src = f'{latest_results_path}/{latest_outputs[0]}'
    
    tensor = torch.load(src)

    print(f'Copy from {src} to {args.dest}')
    np.save(args.dest, tensor.numpy())
    #copy(src, args.dest)