#!/bin/bash
# @ job_name = compute_means
# @ initialdir = /gpfs/scratch/bsc28/bsc28487/tfm_cesc
# @ output= logs/create_TFRecords_logs/compute_means.out
# @ error= logs/create_TFRecords_logs/compute_means.err
# @ total_tasks= 1 
# @ gpus_per_node= 1	
# @ cpus_per_task= 1
# @ features= k80
# @ wall_clock_limit= 00:45:00	

module purge
module load K80/default impi/2018.1 mkl/2018.1 cuda/8.0 CUDNN/7.0.3 python/3.6.3_ML

python compute_means.py --data_dir /gpfs/projects/bsc28/optretina/data --output_dir data/

