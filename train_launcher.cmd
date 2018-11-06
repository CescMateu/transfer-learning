#!/bin/bash
# @ job_name = create_TFRecords
# @ initialdir = /gpfs/scratch/bsc28/bsc28487/tfm_cesc
# @ output= create_TFRecords.out 	
# @ error= create_TFRecords.err 	
# @ total_tasks= 1 
# @ gpus_per_node= 1	
# @ cpus_per_task= 1
# @ features= k80
# @ wall_clock_limit= 00:15:00	

module purge
module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python train.py --model_dir /gpfs/scratch/bsc28/bsc28487/tfm_cesc/experiments/base_model --data_dir /gpfs/scratch/bsc28/bsc28487/tfm_cesc/data/retina_TFRecords