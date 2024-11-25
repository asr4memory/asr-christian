# WER-Results on test-set for eg_fzh-wde_combined_dataset_v1

# CHRISTIAN
| Model | Original | BOHB_jan | multi_GPU | BOHB_feb|
|----------|----------|----------|----------|----------|
| tiny    | 88.15    | 91.54   | 68.48 | |
| small    | 63.23    | 60.14   |      | 35.71 |
| medium   | 60.62    | Data 6   |     | |
| lage-v3 | 27.98 | Data |         |   |

# PETER
| Model | Original | - |
|----------|----------|-|
| tiny    | 88.15    | - |


# INSTRUCTIONS FOR CLUSTER

1. Create a virtual environment with conda and python 3.10 ("conda create --name asr-christian python=3.10"). Install all the packages using within the environment ("pip install -r requirements.txt")


3. Add the data you want to use in the folder [/data/datasets](data/datasets) in your home directory. The files used for training are in *eg_fzh-wde_combined_dataset_v1* which you need to download (on MMT or ask Peter). For convenience, I uploaded a smaller file for testing.


4. Add the [/asr-finetune](asr-finetune) folder into you home directory.

# Useful formulas

total_Gradient_steps = round_up(length_train_set / per_device_train_batch_size) * num_epochs

iterations = round_up(total_Gradient_steps / save_steps)


# Instrucction for Job-Submisison

1. Create a new config-file (copy an existing and paste to the new). Important: change output_tag the outout tag which determines the folder name. Change some settings in # Performance Impacting Settings
2. Create .sh script (copy an existing and paste to the new) for submitting the job. Important: cpus-per-task and gpu must match the requests from the .config file. Don't forget to change the config file at the very end of the .sh script.
3. start job in terminal using sbatch.
4. Monitor jobs with tensorboard:
   - log into the cluster via: ssh -L 16006:127.0.0.1:6007 <your_name@curta.zedat.fu-berlin.de>
   - type in terminal: tensorboard --logdir /ray_results/<output_tag>
   - open browser on your local machine and type: http://127.0.0.1:16006
   - Note that results will based on save_steps. So you need to be a bit patient for things to be seen.
