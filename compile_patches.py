import numpy as np
import torch
from torchvision import transforms
from os import path

from data.util.mask import bbox2mask, tiling_bbox


def pil_loader(path):
    return np.load(path)


#tfs_to_tensor = transforms.Compose([
#    transforms.ToTensor()])

def compile_patches(directory, i, full_size, patch_size, stride, apply_mask=True, until_x=None, until_y=None, until_z=None, bg_fill=float('NaN'), print_warnings=False, permute=False):
    full_img = torch.full((full_size, full_size, full_size), bg_fill, dtype=torch.float32)
    
    for x_start in range(0, full_size - stride, stride):
        for y_start in range(0, full_size - stride, stride):
            for z_start in range(0, full_size - stride, stride):
                
                if until_x != None and until_y != None and until_z != None:
                    if (x_start >= until_x and y_start >= until_y and z_start >= until_z):
                        return full_img

                x_end = x_start + patch_size
                y_end = y_start + patch_size
                z_end = z_start + patch_size

                fname = f'{i}-x-{x_start}-y-{y_start}-z-{z_start}'

                patch_path = f'{directory}/{fname}.npy'

                if not path.isfile(patch_path):
                    if print_warnings:
                        print(f'Skipping {patch_path} because it does not exist!')
                    continue

                patch_img = torch.tensor(pil_loader(patch_path)).squeeze()
                
                if permute: patch_img = patch_img.permute(1, 2, 0)

                if apply_mask:
                    ######################################## Old ########################################
                    # Determine which mask to use
                    # 0: generate full cube | first cube
                    # 1: generate right half | first row after first cube
                    # 2: generate bottom half | first column after first cube
                    # 3: generate right bottom quarter | first plane after first row and after first column
                    # 4: generate back half | first aisle after first cube
                    # 5: generate back right quarter | first row after first aisle and after first plane
                    # 6: generate back bottom quarter | first column after first aisle and after first plane
                    # 7: generate back right bottom eighth | all of the rest

                    # Coordinate system: z into page
                    #   0   y
                    # 0 +--->
                    #   |
                    #   |
                    # x v

                    """mask_type = None
                    if x_start == 0:
                        if y_start == 0:
                            if z_start == 0:
                                mask_type = 0
                            else:
                                mask_type = 4
                        else:
                            if z_start == 0:
                                mask_type = 1
                            else:
                                mask_type = 5
                    else:
                        if y_start == 0:
                            if z_start == 0:
                                mask_type = 2
                            else:
                                mask_type = 6
                        else:
                            if z_start == 0:
                                mask_type = 3
                            else:
                                mask_type = 7"""
                    ###################################### End Old ######################################

                    # Determine which mask to use
                    # 0: generate full cube | first cube
                    # 1: generate right half | first row after first cube
                    # 2: generate right bottom quarter | first plane after first row and after first column
                    # 3: generate back right bottom eighth | all of the rest
                    # 4: generate bottom half | first column after first cube
                    # 5: generate back half | first aisle after first cube
                    # 6: generate back right quarter | first row after first aisle and after first plane
                    # 7: generate back bottom quarter | first column after first aisle and after first plane

                    # Coordinate system: z into page
                    #   0  y
                    # 0 +-->
                    #   |
                    # x v

                    mask_type = None
                    if x_start == 0:
                        if y_start == 0:
                            if z_start == 0:
                                mask_type = 0
                            else:
                                mask_type = 5
                        else:
                            if z_start == 0:
                                mask_type = 1
                            else:
                                mask_type = 6
                    else:
                        if y_start == 0:
                            if z_start == 0:
                                mask_type = 4
                            else:
                                mask_type = 7
                        else:
                            if z_start == 0:
                                mask_type = 2
                            else:
                                mask_type = 3

                    mask = bbox2mask((patch_size, patch_size, patch_size), tiling_bbox(img_shape=(patch_size, patch_size, patch_size), type=mask_type))
                    mask = torch.from_numpy(mask).permute(3, 0, 1, 2)[0]
                else:
                    mask = torch.ones((patch_size, patch_size, patch_size))
                mask = mask.type(torch.bool).reshape((patch_size, patch_size, patch_size))

                # Compute mask that covers the full image
                full_mask = torch.zeros_like(full_img, dtype=bool)
                full_mask[x_start:x_end, y_start:y_end, z_start:z_end] = mask

                full_img[full_mask] = (patch_img[mask]).flatten().float()
    return full_img