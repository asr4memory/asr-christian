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

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == " " ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus 1 --block --temp-dir /scratch/kompiel/tmp_eval &

export RAY_GCS_RPC_TIMEOUT_MS=10000


cd asr-finetune
python -u evaluate_model.py -c configs/eval_whisper_largev3.config 

#python -u evaluate_model.py -c configs/eval_whisper_medium.config --search_schedule_mode large_small_BOHB --model_ckpt_path /home/chrvt/ray_results/whisper_medium_BOHB_jan/TorchTrainer_161cec65_1_learning_rate=0.0001,per_device_train_batch_size=2,warmup_steps=12,weight_decay=0.0000_2024-11-18_17-31-14/checkpoint_000007/checkpoint





