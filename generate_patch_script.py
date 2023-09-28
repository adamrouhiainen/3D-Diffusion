import numpy as np

tile_path = '../matter_density_data_48px_264px_128plr/tile_arepo_seed4443_48px_128p'
index = 0

images = [i for i in np.genfromtxt(f'{tile_path}/flist-{index}.txt', dtype=str, encoding='utf-8')]

cmds = []
for i, image in enumerate(images):
    lr_img_path = f'{tile_path}/lr/{image}.npy'
    sr_img_dir = f'{tile_path}/sr'
    sr_img_path = f'{sr_img_dir}/{image}.npy'
    mask_path = f'{tile_path}/mask/{image}.npy'

    parts = image.split('-')
    x = int(parts[2])
    y = int(parts[4])
    z = int(parts[6])

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
    if x == 0:
        if y == 0:
            if z == 0:
                mask_type = 0
            else:
                mask_type = 5
        else:
            if z == 0:
                mask_type = 1
            else:
                mask_type = 6
    else:
        if y == 0:
            if z == 0:
                mask_type = 4
            else:
                mask_type = 7
        else:
            if z == 0:
                mask_type = 2
            else:
                mask_type = 3

    single_dataset_name = 'single-datasets/single-tile'

    cmds.append(f'python generate_mask.py --index {index} --dir {sr_img_dir} -o {mask_path} -x {x} -y {y} -z {z}')
    cmds.append(f'python prepare_single_dataset.py --fname {image} --lr-img-path {lr_img_path} --mask-img-path {mask_path} --mask-type {mask_type} --out-dir {single_dataset_name}')
    cmds.append(f'python run.py -p test -c {single_dataset_name}/config.json')
    #cmds.append(f'python run.py -p test --gpu \'1\' -c {single_dataset_name}/config.json')
    cmds.append(f'python copy-latest-output.py -d {sr_img_path} -f test_superresolution-single_')
    cmds.append('')

np.savetxt('run_big.bat', np.array(cmds, dtype=str), '%s', encoding='utf-8')