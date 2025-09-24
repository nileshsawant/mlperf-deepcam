#!/bin/bash
#SBATCH -A esifapps
#SBATCH -J train_cam5
#SBATCH -t 00:10:00
##SBATCH -p gpu-h100
##SBATCH -p debug
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=80G
##SBATCH -n 32
##SBATCH --exclusive
#SBATCH --gres=gpu:1

# The MIT License (MIT)
#
# Copyright (c) 2020 NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#load stuff
##module load cuda/12.3
module load anaconda3/2024.06.1
#module load PrgEnv-nvhpc/8.5.0
module load nccl/2.21.5
##module load craype-network-ucx
##module load cray-mpich-ucx/8.1.28
##export MPICH_RDMA_ENABLED_CUDA=1
##export MPICH_GPU_SUPPORT_ENABLED=1
##export MPICH_OFI_NIC_POLICY=GPU
##export MPICH_GPU_SUPPORT_ENABLED=0

conda activate /projects/hpcapps/nsawant/torchParallelMPIdeepcam
##export PROJ_LIB=/projects/hpcapps/nsawant/torchParallelMPIdeepcam/share/basemap
export PYTHONPATH=/projects/hpcapps/nsawant/torchParallelMPIdeepcam/lib/python3.13/site-packages:${PYTHONPATH}

#ranks per node
rankspernode=1
totalranks=$(( ${SLURM_NNODES} * ${rankspernode} ))

#parameters
run_tag="deepcam_prediction_run1-kestrel"
data_dir_prefix="/scratch/nsawant/deepcam/All-Hist"
output_dir="/scratch/nsawant/deepcam/cam5_runs/${run_tag}"

#create files
mkdir -p ${output_dir}
touch ${output_dir}/train.out

#run training
srun --overlap -u -N ${SLURM_NNODES} -n ${totalranks} -c $(( 32 / ${rankspernode} )) --cpu_bind=cores --gpu-bind=single:1 \
     python ../train_hdf5_ddp.py \
     --wireup_method "nccl-slurm-pmi" \
     --run_tag ${run_tag} \
     --data_dir_prefix ${data_dir_prefix} \
     --output_dir ${output_dir} \
     --max_inter_threads 0 \
     --model_prefix "classifier" \
     --optimizer "AdamW" \
     --start_lr 1e-3 \
     --lr_schedule type="multistep",milestones="15000 25000",decay_rate="0.1" \
     --lr_warmup_steps 0 \
     --lr_warmup_factor $(( ${SLURM_NNODES} / 8 )) \
     --weight_decay 1e-2 \
     --validation_frequency 200 \
     --training_visualization_frequency 200 \
     --validation_visualization_frequency 40 \
     --max_validation_steps 50 \
     --logging_frequency 0 \
     --save_frequency 400 \
     --max_epochs 200 \
     --amp_opt_level O1 \
     --wandb_certdir /scratch/nsawant/deepcam \
     --local_batch_size 2 |& tee -a ${output_dir}/train.out
