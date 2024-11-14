#!/bin/bash

#SBATCH --mail-user=<chrvt@zedat.fu-berlin.de>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="evaluate_small"
#SBATCH --time=02:00:00
#SBATCH --mem=64G  #32

#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --tasks-per-node=1  ### ensure that each Ray worker runtime will run on a separate node
#SBATCH --cpus-per-task=4  ### cpus and gpus per node 
#SBATCH --gres=gpu:1

###SBATCH --mem-per-cpu=1GB

#SBATCH --partition=gpu
#SBATCH --qos=standard


###SBATCH --nodelist=g007


module load cuDNN/8.4.1.50-CUDA-11.7.0
nvidia-smi
nvcc --version


cd asr-christian
python -u evaluate_model.py -c configs/eval_whisper_small.config

# --search_schedule_mode large_small_BOHB --model_ckpt_path /home/chrvt/ray_results/whisper_medium_BOHB_MVP/TorchTrainer_fb57dfe6_15_learning_rate=0.0000,per_device_train_batch_size=16,warmup_steps=30,weight_decay=0.0717_2024-11-10_17-56-05/checkpoint_000000/checkpoint


#/home/chrvt/ray_results/whisper_small_BOHB/TorchTrainer_3084de0e_4_learning_rate=0.0000,per_device_train_batch_size=16,warmup_steps=38,weight_decay=0.1002_2024-11-11_21-33-21/checkpoint_000000/checkpoint



