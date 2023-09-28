# Copyright (c) OpenMMLab. All rights reserved.
import math
import numpy as np
from PIL import Image, ImageDraw


def random_cropping_bbox(img_shape=(256,256), mask_mode='onedirection'):
    h, w = img_shape
    if mask_mode == 'onedirection':
        _type = np.random.randint(0, 4)
        if _type == 0:
            top, left, height, width = 0, 0, h, w//2
        elif _type == 1:
            top, left, height, width = 0, 0, h//2, w
        elif _type == 2:
            top, left, height, width = h//2, 0, h//2, w
        elif _type == 3:
            top, left, height, width = 0, w//2, h, w//2
    else:
        target_area = (h*w)//2
        width = np.random.randint(target_area//h, w)
        height = target_area//width
        if h==height:
            top = 0
        else:
            top = np.random.randint(0, h-height)
        if w==width:
            left = 0
        else:
            left = np.random.randint(0, w-width)
    return (top, left, height, width)

def random_bbox(img_shape=(256,256), max_bbox_shape=(128, 128), max_bbox_delta=40, min_margin=20):
    """Generate a random bbox for the mask on a given image.

    In our implementation, the max value cannot be obtained since we use
    `np.random.randint`. And this may be different with other standard scripts
    in the community.

    Args:
        img_shape (tuple[int]): The size of a image, in the form of (h, w).
        max_bbox_shape (int | tuple[int]): Maximum shape of the mask box,
            in the form of (h, w). If it is an integer, the mask box will be
            square.
        max_bbox_delta (int | tuple[int]): Maximum delta of the mask box,
            in the form of (delta_h, delta_w). If it is an integer, delta_h
            and delta_w will be the same. Mask shape will be randomly sampled
            from the range of `max_bbox_shape - max_bbox_delta` and
            `max_bbox_shape`. Default: (40, 40).
        min_margin (int | tuple[int]): The minimum margin size from the
            edges of mask box to the image boarder, in the form of
            (margin_h, margin_w). If it is an integer, margin_h and margin_w
            will be the same. Default: (20, 20).

    Returns:
        tuple[int]: The generated box, (top, left, h, w).
    """
    if not isinstance(max_bbox_shape, tuple):
        max_bbox_shape = (max_bbox_shape, max_bbox_shape)
    if not isinstance(max_bbox_delta, tuple):
        max_bbox_delta = (max_bbox_delta, max_bbox_delta)
    if not isinstance(min_margin, tuple):
        min_margin = (min_margin, min_margin)

    img_h, img_w = img_shape[:2]
    max_mask_h, max_mask_w = max_bbox_shape
    max_delta_h, max_delta_w = max_bbox_delta
    margin_h, margin_w = min_margin

    if max_mask_h > img_h or max_mask_w > img_w:
        raise ValueError(f'mask shape {max_bbox_shape} should be smaller than '
                         f'image shape {img_shape}')
    if (max_delta_h // 2 * 2 >= max_mask_h
            or max_delta_w // 2 * 2 >= max_mask_w):
        raise ValueError(f'mask delta {max_bbox_delta} should be smaller than'
                         f'mask shape {max_bbox_shape}')
    if img_h - max_mask_h < 2 * margin_h or img_w - max_mask_w < 2 * margin_w:
        raise ValueError(f'Margin {min_margin} cannot be satisfied for img'
                         f'shape {img_shape} and mask shape {max_bbox_shape}')

    # get the max value of (top, left)
    max_top = img_h - margin_h - max_mask_h
    max_left = img_w - margin_w - max_mask_w
    # randomly select a (top, left)
    top = np.random.randint(margin_h, max_top)
    left = np.random.randint(margin_w, max_left)
    # randomly shrink the shape of mask box according to `max_bbox_delta`
    # the center of box is fixed
    delta_top = np.random.randint(0, max_delta_h // 2 + 1)
    delta_left = np.random.randint(0, max_delta_w // 2 + 1)
    top = top + delta_top
    left = left + delta_left
    h = max_mask_h - delta_top
    w = max_mask_w - delta_left
    return (top, left, h, w)


def tiling_bbox_old(img_shape=(48, 48, 48), type=None):
    h, w, d = img_shape
    _type = np.random.randint(0, 4) if type == None else type
    if _type == 0:
        # Mask everything
        top, left, bottom, height, width, depth = 0, 0, 0, h, w, d
    elif _type == 1:
        # Draw full left
        top, left, bottom, height, width, depth = 0, w//2, 0, h, w//2, d
    #elif _type == 2:
    #   # Draw full bottom
    #    top, left, height, width = h//2, 0, h//2, w
    elif _type == 2:
        # Draw edge (like an L)
        top, left, bottom, height, width, depth = h//2, w//2, 0, h//2, w//2, d
    elif _type == 3:
        # Draw corner
        top, left, bottom, height, width, depth = h//2, w//2, d//2, h//2, w//2, d//2
    return (top, left, bottom, height, width, depth)


def tiling_bbox(img_shape, type=None): #img_shape=(48, 48, 48)
    """
    Determine which mask to use
    0: generate full cube | first cube
    1: generate right half | first row after first cube
    2: generate right bottom quarter | first plane after first row and after first column
    3: generate back right bottom eighth | all of the rest
    4: generate bottom half | first column after first cube
    5: generate back half | first aisle after first cube
    6: generate back right quarter | first row after first aisle and after first plane
    7: generate back bottom quarter | first column after first aisle and after first plane

    Coordinate system: z into page
      0  y
    0 +-->
      |
    x v

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
    """
                
    h, w, d = img_shape
    _type = np.random.randint(0, 8) if type == None else type
    if _type == 0:
        # Mask everything
        top, left, bottom, height, width, depth = 0, 0, 0, h, w, d
    elif _type == 1:
        # Draw full left
        top, left, bottom, height, width, depth = 0, w//2, 0, h, w//2, d
    elif _type == 2:
        # Draw edge (like an L)
        top, left, bottom, height, width, depth = h//2, w//2, 0, h//2, w//2, d
    elif _type == 3:
        # Draw corner
        top, left, bottom, height, width, depth = h//2, w//2, d//2, h//2, w//2, d//2
    elif _type == 4:
        # Draw full top
        top, left, bottom, height, width, depth = h//2, 0, 0, h//2, w, d
    elif _type == 5:
        # Draw full front
        top, left, bottom, height, width, depth = 0, 0, d//2, h, w, d//2
    elif _type == 6:
        # Draw L
        top, left, bottom, height, width, depth = 0, w//2, d//2, h, w//2, d//2
    elif _type == 7:
        # Draw L
        top, left, bottom, height, width, depth = h//2, 0, d//2, h//2, w, d//2
    return (top, left, bottom, height, width, depth)


def bbox2mask(img_shape, bbox, dtype='uint8'):
    """Generate mask in ndarray from bbox.

    The returned mask has the shape of (h, w, d, 1). '1' indicates the
    hole and '0' indicates the valid regions.

    We prefer to use `uint8` as the data type of masks, which may be different
    from other codes in the community.

    Args:
        img_shape (tuple[int]): The size of the image.
        bbox (tuple[int]): Configuration tuple, (top, left, bottom, height, width, depth)
        dtype (str): Indicate the data type of returned masks. Default: 'uint8'

    Return:
        numpy.ndarray: Mask in the shape of (h, w, d, 1).
    """

    height, width, depth = img_shape[:3]

    mask = np.zeros((height, width, depth, 1), dtype=dtype)
    mask[bbox[0]:bbox[0] + bbox[3], bbox[1]:bbox[1] + bbox[4], bbox[2]:bbox[2] + bbox[5], :] = 1

    return mask
