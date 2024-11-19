#!/bin/bash

#SBATCH --mail-user=<chrvt@zedat.fu-berlin.de>
#SBATCH --mail-type=fail,end
#SBATCH --job-name="evaluate_large"
#SBATCH --time=04:00:00
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


###module load cuDNN/8.4.1.50-CUDA-11.7.0
module load CUDA/12.0.0
nvidia-smi
nvcc --version


cd asr-finetune
python -u evaluate_model.py -c configs/eval_whisper_largev3.config 

#python -u evaluate_model.py -c configs/eval_whisper_medium.config --search_schedule_mode large_small_BOHB --model_ckpt_path /home/chrvt/ray_results/whisper_medium_BOHB_jan/TorchTrainer_161cec65_1_learning_rate=0.0001,per_device_train_batch_size=2,warmup_steps=12,weight_decay=0.0000_2024-11-18_17-31-14/checkpoint_000007/checkpoint





