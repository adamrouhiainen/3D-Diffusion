# 3D-Diffusion
Anonymous 3D diffusion for cosmology

To train the model on (say) GPUs 0 and 1:

python run.py -c 'config/my_config_file.json' --gpu '0,1'

For iterative outpainting, "python run.py" is called many times over, each time with a new .json file shifting which data is run on. To automate this, first generate a batch file with

python generate_patch_script.py

Then, run

sh run_big.bat

The data that the outpainting .json file looks for finds must already be properly generated in the outpanting invervals that you want (we use 24px intervals on 48px cubes).
