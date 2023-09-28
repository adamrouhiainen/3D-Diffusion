import argparse
from collections import OrderedDict
import json
from os import mkdir, path
from shutil import copy, rmtree
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', '--fname', type=str, default=None, help="Path to low res image")
    parser.add_argument('--lr-img-path',  type=str, default=None, help="Path to low res image")
    parser.add_argument('--mask-img-path',  type=str, default=None, help="Path to mask image")
    parser.add_argument('--mask-type',  type=int, default=0, help="Mask type")
    parser.add_argument('-c', '--config', type=str, default='config/sr_mask_arepo_48px_264px_128p.json', help='JSON file for configuration')
    parser.add_argument('-o', '--out-dir', type=str, default=f'single-datasets/{int(time.time())}', help='Dataset output directory')

    args = parser.parse_args()

    print(args)

    """
    Copy over files
    """

    # if path.isdir(args.out_dir):
    #     raise f'{args.out_dir} already exists!'
    if path.isdir(args.out_dir):
        rmtree(args.out_dir)

    mkdir(args.out_dir)

    hr_path = f'{args.out_dir}/hr'
    mkdir(hr_path)

    lr_path = f'{args.out_dir}/lr'
    mkdir(lr_path)

    copy(args.lr_img_path, f'{lr_path}/{args.fname}.npy')
    copy(args.mask_img_path, f'{hr_path}/{args.fname}.npy')

    """
    Write flist.txt
    """
    flist_path = f'{args.out_dir}/flist.txt'
    f = open(flist_path, 'w')
    f.write(args.fname)
    f.close()

    """
    Parse existing config, modify to specify single dataset
    """
    json_str = ''
    with open(args.config, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    config = json.loads(json_str, object_pairs_hook=OrderedDict)

    # Modify with what we want
    config['name'] = 'superresolution-single'
    config['datasets']['test']['which_dataset']['args']['data_root'] = args.out_dir
    config['datasets']['test']['which_dataset']['args']['data_flist'] = flist_path
    config['datasets']['test']['which_dataset']['args']['mask_config']['tiling_mode'] = args.mask_type

    with open(f'{args.out_dir}/config.json', 'w') as f:
        json.dump(config, f, indent=4)
