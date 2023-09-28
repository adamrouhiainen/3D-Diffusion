import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch

from compile_patches import compile_patches

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int, default=0, help="Index of images")
    parser.add_argument('-d', '--dir', type=str, default='patches', help="Directory containing images")
    parser.add_argument('-o', '--output', type=str, default=None, help="Path to mask output")
    parser.add_argument('-x', type=int, default=None, help="x-start of desired generated HR image")
    parser.add_argument('-y', type=int, default=None, help="y-start of desired generated HR image")
    parser.add_argument('-z', type=int, default=None, help="z-start of desired generated HR image")
    parser.add_argument('-f', '--full-size', type=int, default=264, help="Width and height of full image")
    parser.add_argument('-p', '--patch-size', type=int, default=48, help="Width and height of a patch")
    parser.add_argument('-s', '--stride', type=int, default=(48//2), help="Stride between patches")

    ''' parser configs '''
    args = parser.parse_args()

    print(args)
    # bg_full=0 for 3D data instead of old bg_fill=1 for 2D data (due to normalization differences from originally using PIL in 2D)
    full_img = compile_patches(args.dir, args.index, args.full_size, args.patch_size, args.stride, apply_mask=True, until_x=args.x, until_y=args.y, until_z=args.z, bg_fill=0)
    
    #torch.save(full_img, args.output + 'full_img_test')

    if args.x != None and args.y != None and args.z != None:
        x_start = args.x
        y_start = args.y
        z_start = args.z
        x_end = x_start + args.patch_size
        y_end = y_start + args.patch_size
        z_end = z_start + args.patch_size

        # print('min', torch.amin(full_img))
        # print('max', torch.amax(full_img))

        #img = (full_img[x_start:x_end, y_start:y_end, z_start:z_end]).type(torch.uint8).numpy()
        img = (full_img[x_start:x_end, y_start:y_end, z_start:z_end]).numpy()
        np.save(args.output, img)