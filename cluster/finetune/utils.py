""" Collection of utility functions/classes for pre-processing data, saving and more.

Functions are:
   save_file
   load_and_prepare_data_from_folders
   normalize
   steps_per_epoch

Classes are:
    DataCollatorSpeechSeq2SeqWithPadding
"""
import pdb
import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
import json
import logging
import re
logger = logging.getLogger(__name__)

def save_file(file,output_dir,mode='config',file_tag = ''):
    """Saves {config,eval_results} files.

    Args:
        file (txt,json): A text or json file to be saved.
        output_dir (str): Path to output directory where file will be stored
        mode (str): If `config`: saves config file. If `eval_results`: saves the output eval results as json.
    """
    if mode == 'config':
        config_path = os.path.join(output_dir, file_tag + 'config.txt')
        with open(config_path, 'a') as f:
            print(file, file=f)

    elif mode == 'json':
        eval_path = os.path.join(output_dir, file_tag + '.json')
        with open(eval_path, 'w') as f:
            json.dump(file, f)

def load_and_prepare_data_from_folders(path,feature_extractor,tokenizer,test_size=0.2, seed = 0, evaluate = False, debug = False):
    """Loads and prepares data from a folder directory.

    `Important`: Each folder needs to have a subfolder "data" (name not important) containing the .mav audio files AND
                 a metadata.csv file with columns 'file_name' and 'transcription'. The file_name must match the file
                 name in the data folder. The transcription is a string of the true transcription of the audio.

    We
        1. loop through subfolders of path-folder, load each folder as dataset, and concatenate datasets
        2. Do the train, validation, and test splits
        3. Resample the audio data and compute the log-Mel input features

    Args:
        path (str): Directory path of head data folder
        feature_extractor (WhisperFeatureExtractor): Feature extractor calculates the log-Mel representations of inputs
                                                     used for training and evaluation
        tokenizer (WhisperTokenizer): The tokenizer is converting target-text into tokens.
        test_size (float): Fraction of total data used for testing.
        seed (int): random seed for reproducibility
        evaluate (bool): If true, only the test set is created. Otherwise: train and validation set.
        debug (bool): If true, does some statistics on the test set (if evaluate=True) or validation set (otherwise).
                      Should result to the same value if one wants to compare two different models.
    """
    data_collection = []
    first_level_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    num_rows = 0
    """Step 1"""
    for subfolder in first_level_subfolders:
        dataset = load_dataset("audiofolder", data_dir=subfolder) # Laden des Datasets von der bereinigten CSV-Datei
        data_collection += [dataset['train']]
        num_rows += dataset['train'].num_rows

    dataset = concatenate_datasets(data_collection)

    assert dataset.num_rows == num_rows, "Some data got lost in the process of concatenation."

    """Step 2: Dataset in Trainings- und Testsets aufteilen """
    split_dataset = dataset.train_test_split(test_size=test_size, seed = seed)  # 20% für Testdaten
    split_trainset = split_dataset['train'].train_test_split(test_size=0.1, seed = seed) # 10% of training for validation

    # Erstellen eines DatasetDict-Objekts
    if evaluate:
        dataset_dict = DatasetDict({
            'test': split_dataset['test']
        })
    else:
        dataset_dict = DatasetDict({
            'train': split_trainset['train'], #split_dataset['train'],
            'validation': split_trainset['test'],
        })

    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    """Step 3: Apply prepare_dataset mapping to dataset_dict"""
    dataset_dict = dataset_dict.map(prepare_dataset,
                                    remove_columns=dataset_dict.column_names["test"] if evaluate else dataset_dict.column_names["train"], num_proc=1)

    logger.info('len validation set: %s', split_trainset['test'].num_rows)
    logger.info('len test set: %s', split_dataset['test'].num_rows)

    # if debug:
    #     """Do some statistics for comparison to ensure correct splits were performed. Alternatively: just
    #        compare the .json eval outputs."""
    #     data_ = 'test' if evaluate else 'validation'
    #     logger.info('Sum of first 3 %s examples divided by total number of %s examples: %.2f',data_,data_,
    #             (sum(dataset_dict[data_]['input_features'][0][0]) +
    #                   sum(dataset_dict[data_]['input_features'][0][1]) +
    #                   sum(dataset_dict[data_]['input_features'][0][2])
    #                    )/dataset_dict[data_].num_rows
    #             )
    return dataset_dict, split_trainset['train'].num_rows


from dataclasses import dataclass
from typing import Any
import numpy as np
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Data Collator for Speech Seq2Seq models with Padding

    We used the collator suggested by the tutorial: https://huggingface.co/blog/fine-tune-whisper.
    We had to slightly modify it due our data being in a different format as required by ray tune.

    Attributes:
       processor (WhisperProcessor): Processor used for padding (normalizing data to same length)
       decoder_start_token_id (int): Token indicating the Beginning Of Setence (BOS)

    Methods:
       __call__(features): Processing a dictionary of input features
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features):
        """ Processing a dictionary of input features.

        Input features are padded to `longest` forms and pytorch tensors are returned.

        Args:
            features (dict): A dictionary with keys 'input_features' consiting of log-Mel features and tokenized
                             'labels'
        Returns"
            batch (dict): Dictionary of padded `input_features` and `labels` as pytorch tensors
        """
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": np.vstack(list(feature))} for feature in features["input_features"]]
        batch = self.processor.feature_extractor.pad(input_features,  padding='longest', return_tensors="pt")

        lab_feat = [{"input_ids": feature} for feature in features["labels"]]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(lab_feat, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def normalize(text):
    """
    Removes certain characters from text and lowers cases.

    Args:
        text (str or list of str): Single string or list of strings to be normalized.

    Returns:
        str or list of str: Normalized string or list of normalized strings.
    """
    def process_single_text(single_text):
        result = single_text.strip().lower()
        result = re.sub(r"[!\?\.,;]", "", result)
        return result

    if isinstance(text, list):
        return [process_single_text(t) for t in text]
    elif isinstance(text, str):
        return process_single_text(text)
    else:
        raise TypeError("Input must be a string or a list of strings.")


def steps_per_epoch(len_train_set,batch_size):
    """Calculates the total number of gradient steps

    Assume gradient_accumulation_steps = 1.

    TODO:
        * Add gradient_accumulation_steps > 1
        * adjust train.py to allow for gradient accumulations

    Args:
        len_train_set (int): Total dataset length
        batch_size (int): batch size
    """
    if len_train_set % batch_size == 0:
        return int(len_train_set / batch_size)
    else:
       return int(len_train_set / batch_size) + 1

# Define a custom argument type for a list of strings
def list_of_strings(arg):
    return arg.split(',')