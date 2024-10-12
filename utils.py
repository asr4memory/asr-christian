# Collection of utility functions for pre-processing
import pdb

from datasets import load_dataset, DatasetDict


def load_and_prepare_data_from_folder(path,feature_extractor,tokenizer,test_size=0.2):
    # Laden des Datasets von der bereinigten CSV-Datei
    dataset = load_dataset("audiofolder", data_dir=path)

    # pdb.set_trace()
    # Dataset in Trainings- und Testsets aufteilen
    split_dataset = dataset['train'].train_test_split(test_size=test_size)  # 20% für Testdaten

    split_trainset = split_dataset['train'].train_test_split(test_size=0.1) # 10% of training for validation

    # Erstellen eines DatasetDict-Objekts
    dataset_dict = DatasetDict({
        'train': split_trainset['train'], #split_dataset['train'],
        'validation': split_trainset['test'],
        'test': split_dataset['test']
    })

    # prepare dataset: 1. map into log-Mel representation 2. downsample to correct sampling rate 3. map targets to readable token-format
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] =  feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    dataset_dict = dataset_dict.map(prepare_dataset, remove_columns=dataset_dict.column_names["train"], num_proc=1)

    return dataset_dict

import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch
