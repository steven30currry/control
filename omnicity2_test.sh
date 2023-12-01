#!/usr/bin/bash

export CUDA_HOME=/mnt/petrelfs/share/cuda-11.3/
export PATH=/mnt/petrelfs/share/cuda-11.3/bin:$PATH
export LD_LIBRARY_PATH=/mnt/petrelfs/share/cuda-11.3/lib64:$LD_LIBRARY_PATH

source ~/miniconda3/etc/profile.d/conda.sh
conda activate control
export PYTHONPATH=.:$PYTHONPATH

# mirror
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=8 \
#     --config_path=./models/cldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202310111653/image_log/test \
#     --model_path=results/202310111653/epoch=57-step=13281.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/mirror/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=8 \
#     --config_path=./models/lacldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202310122129/image_log/test \
#     --model_path=results/202310122129/epoch=93-step=21525.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/mirror/random-cleaned2-test.csv

# geo_height
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=8 \
#     --config_path=./models/cldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202310111703/image_log/test \
#     --model_path=results/202310111703/epoch=28-step=6640.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=./models/cldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202310161536/image_log/test \
#     --model_path=results/202310161536/epoch=15-step=3663.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/geo_height/random-cleaned2-test.csv
# old
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=8 \
#     --config_path=./models/cldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202308040919/image_log/test \
#     --model_path=results/202308040919/epoch=85-step=5675.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity/medium-view/cleaned_geo-height_test.csv

# polar_sate
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=8 \
#     --config_path=./models/cldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202310111857/image_log/test \
#     --model_path=results/202310111857/epoch=28-step=6640.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=./models/cldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202310161538/image_log/test \
#     --model_path=results/202310161538/epoch=15-step=3663.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
    --num_gpus=4 \
    --config_path=./models/cldm_v15.yaml \
    --image_width=1024 --image_height=512 \
    --result_dir=results/202310171130/image_log/test \
    --model_path=results/202310171130/epoch=57-step=13281.ckpt \
    --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=8 \
#     --config_path=./models/cldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --text_prompt='' --unconditional_guidance_scale=1.0 \
#     --result_dir=results/202310120245/image_log/test \
#     --model_path=results/202310120245/epoch=28-step=6640.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=8 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=8 \
#     --config_path=./models/lacldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --text_prompt='' --unconditional_guidance_scale=1.0 \
#     --result_dir=results/202310121022/image_log/test \
#     --model_path=results/202310121022/epoch=28-step=6640.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
# srun -p bigdata --async --nodes=1 --ntasks=4 --gpus-per-task=1 --cpus-per-task=4 python omnicity_test.py \
#     --num_gpus=4 \
#     --config_path=./models/lacldm_v15.yaml \
#     --image_width=1024 --image_height=512 \
#     --result_dir=results/202310162222/image_log/test \
#     --model_path=results/202310162222/epoch=97-step=22441.ckpt \
#     --data_file_path=/mnt/petrelfs/share_data/zhonghuaping.p/datasets/OmniCity2/polar_sate/random-cleaned2-test.csv
