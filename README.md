# [3D-Diffusion for Super-resolution Emulation in Cosmology](https://arxiv.org/pdf/2311.05217.pdf)

![Denoising diffusion for 3D fields](https://github.com/adamrouhiainen/3D-Diffusion/blob/main/diffusion_example.png)

The model is described in https://arxiv.org/abs/2311.05217

Our code is built off of https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models.

Requirements:  `torch>=1.6`, `numpy`, `scipy`, `tqdm`

To train the model on (say) GPUs 0 and 1:

`python run.py -c 'config/sr-mask-arepo-48px-264px-128p-2000T.json' --gpu '0,1'`

To generate a field from a trained model, change the phase to test:

`python run.py -c 'config/sr-mask-arepo-48px-264px-128p-2000T.json' -p test`

To run the iterative outpainting described in the [paper](https://arxiv.org/abs/2311.05217), `python run.py` is called many times over, each time with a new .json file shifting which data is run on. To automate this, first generate a batch file `run_big.bat` with

`python generate_patch_script.py`

Then, run

`sh run_big.bat`

For iterative outpainting, the low-resolution data that the .json file finds must be properly organized sequentially in the outpanting invervals that you want (we used 24px intervals on 48px cubes). There are scripts to preprocess the data in `matter_density_data/`.
