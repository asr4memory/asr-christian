# WER-Results on test-set

| Model | Original | BOHB_jan |
|----------|----------|----------|
| tiny    | 88.15    | Data 2   |
| small    | 63.23    | Data 4   |
| medium   | 60.62    | Data 6   |


# INSTRUCTIONS FOR CLUSTER

1. Create a virtual environment and install the [requirements.txt](requirements.txt). 
   I used mini-conda as my virtual environment manager: conda create --name <env> --file requirements.txt
   With or using pip only: pip install -r requirements.txt within the environment


3. Add the data you want to use in the folder [/data/datasets](data/datasets) in your home directory. The files used for training are in *eg_fzh-wde_combined_dataset_v1* which you need to download (on MMT or ask Peter). For convenience, I uploaded a smaller file for testing.


4. Add the [/asr-finetune](asr-finetune) folder into you home directory.
