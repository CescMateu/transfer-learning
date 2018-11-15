#!/bin/bash
# @ job_name = architecture_HPAI
# @ initialdir = /gpfs/scratch/bsc28/bsc28487/tfm_cesc
# @ output= logs/architecture_HPAI/architecture_HPAI.out
# @ error= logs/architecture_HPAI/architecture_HPAI.err
# @ total_tasks= 1 
# @ gpus_per_node= 1	
# @ cpus_per_task= 1
# @ features= k80
# @ wall_clock_limit= 04:30:00

module purge
module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python train.py --model_dir experiments/architecture_HPAI --data_dir data/retina_TFRecords --preproc_file data/mean_channels_img.npy