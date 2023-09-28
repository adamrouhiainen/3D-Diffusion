# 3D-Diffusion
Anonymous 3D diffusion for cosmology

To train the model on (say) GPUs 0 and 1:
python run.py -c 'config/my_config_file.json' --gpu '0,1'

For iterative outpainting, first generate the "patch script" with
python generate_patch_script.py
Then, run
sh run_big.bat
