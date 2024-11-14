# Collection of utility functions for pre-processing
import pdb
import os
from datasets import load_dataset, DatasetDict, concatenate_datasets
import json
import logging
logger = logging.getLogger(__name__)

def save_file(file,output_dir,mode='config',file_tag = ''):

    if mode == 'config':
        config_path = os.path.join(output_dir, file_tag + 'config.txt')
        with open(config_path, 'a') as f:
            print(file, file=f)
        # if not os.path.exists(config_path):
        #     # os.makedirs(os.path.join(os.getcwd(),args.output_dir))

    elif mode == 'eval_results':
        eval_path = os.path.join(output_dir, file_tag + '.txt')
        with open(eval_path, 'a') as f:
            print(file, file=f)

    elif mode == 'json':
        eval_path = os.path.join(output_dir, file_tag + '.json')
        with open(eval_path, 'w') as f:
            json.dump(file, f)

def load_and_prepare_data_from_folder(path,feature_extractor,tokenizer,test_size=0.2, seed = 0):
    # Laden des Datasets von der bereinigten CSV-Datei
    dataset = load_dataset("audiofolder", data_dir=path)

    # Dataset in Trainings- und Testsets aufteilen
    split_dataset = dataset['train'].train_test_split(test_size=test_size, seed = seed)   # 20% für Testdaten

    split_trainset = split_dataset['train'].train_test_split(test_size=0.1, seed = seed)  # 10% of training for validation

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
        # import pdb; pdb.set_trace()
        # encode target text to label ids
        batch["labels"] = tokenizer(batch["transcription"]).input_ids
        return batch

    dataset_dict = dataset_dict.map(prepare_dataset, remove_columns=dataset_dict.column_names["train"], num_proc=1)

    logger.info('len validation set: %s', split_trainset['test'].num_rows)
    logger.info('len test set: %s', split_dataset['test'].num_rows)


    # logger.info('average of first 5 test examples: %s',
    #             (sum(dataset_dict['test']['input_features'][0][0]) +
    #                   sum(dataset_dict['test']['input_features'][0][1]) +
    #                   sum(dataset_dict['test']['input_features'][0][2])
    #                    )/split_dataset['test'].num_rows
    #             )

    return dataset_dict, split_trainset['train'].num_rows

def load_and_prepare_data_from_folders(path,feature_extractor,tokenizer,test_size=0.2, seed = 0):
    data_collection = []
    # 1. loop through subfolders of path-folder 2. load each folder as dataset 3. concatenate datasets

    first_level_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    num_rows = 0
    for subfolder in first_level_subfolders:
        dataset = load_dataset("audiofolder", data_dir=subfolder) # Laden des Datasets von der bereinigten CSV-Datei
        data_collection += [dataset['train']]
        num_rows += dataset['train'].num_rows

    dataset = concatenate_datasets(data_collection)

    assert dataset.num_rows == num_rows, "Some data got lost in the process of concatenation."

    # Dataset in Trainings- und Testsets aufteilen
    split_dataset = dataset.train_test_split(test_size=test_size, seed = seed)  # 20% für Testdaten

    split_trainset = split_dataset['train'].train_test_split(test_size=0.1, seed = seed) # 10% of training for validation

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

    logger.info('len validation set: %s', split_trainset['test'].num_rows)
    logger.info('len test set: %s', split_dataset['test'].num_rows)

    # logger.info('average of first 5 test examples: %s',
    #             (sum(dataset_dict['test']['input_features'][0][0]) +
    #                   sum(dataset_dict['test']['input_features'][0][1]) +
    #                   sum(dataset_dict['test']['input_features'][0][2])
    #                    )/split_dataset['test'].num_rows
    #             )

    return dataset_dict, split_trainset['train'].num_rows

def load_and_prepare_data_for_eval(path,feature_extractor,tokenizer,test_size=0.2, seed = 0):
    data_collection = []
    # 1. loop through subfolders of path-folder 2. load each folder as dataset 3. concatenate datasets

    first_level_subfolders = [f.path for f in os.scandir(path) if f.is_dir()]
    num_rows = 0
    for subfolder in first_level_subfolders:
        dataset = load_dataset("audiofolder", data_dir=subfolder) # Laden des Datasets von der bereinigten CSV-Datei
        data_collection += [dataset['train']]
        num_rows += dataset['train'].num_rows

    dataset = concatenate_datasets(data_collection)

    assert dataset.num_rows == num_rows, "Some data got lost in the process of concatenation."

    # Dataset in Trainings- und Testsets aufteilen
    split_dataset = dataset.train_test_split(test_size=test_size, seed = seed)  # 20% für Testdaten

    split_trainset = split_dataset['train'].train_test_split(test_size=0.1, seed = seed) # 10% of training for validation

    # Erstellen eines DatasetDict-Objekts
    dataset_dict = DatasetDict({
        # 'train': split_trainset['train'], #split_dataset['train'],
        # 'validation': split_trainset['test'],
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

    dataset_dict = dataset_dict.map(prepare_dataset, num_proc=1)

    logger.info('average of first 5 test examples: $s',
                (sum(dataset_dict['test']['input_features'][0][0]) +
                      sum(dataset_dict['test']['input_features'][0][1]) +
                      sum(dataset_dict['test']['input_features'][0][2])
                       )/split_dataset['test'].num_rows
                )

    return dataset_dict, split_trainset['train'].num_rows


from torch.utils.data import Dataset
class CustomAudioDataset(Dataset):
    def __init__(self, dataset, collate_fn):
        self.dataset = dataset
        self.collate_fn = collate_fn

    def __len__(self):
        return self.dataset.num_rows

    def __getitem__(self, idx):
        # pdb.set_trace()
        # datum = self.collate_fn(self.dataset.__getitem__(idx))
        output = self.collate_fn(self.dataset.__getitem__(idx))
        return output["input_features"], output["labels"]
            #self.collate_fn(self.dataset.__getitem__(idx)))



import torch

from dataclasses import dataclass
from typing import Any, Dict, List, Union
import numpy as np
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features): # : List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        # input_features = features["input_features"]
        # input_features = features["input_features"] # [{"input_features": feature["input_features"]} for feature in features]
        import pdb;

        input_features = [{"input_features": np.vstack(list(feature))} for feature in features["input_features"]]

        # for feature in features["input_features"]: print(feature)

        # for feature in features["input_features"]:
        #     for feat in feature: print(feat.shape)
        # pdb.set_trace()
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        # batch = self.processor.feature_extractor.pad(list(input_features), return_tensors="pt")
        # get the tokenized label sequences

        # label_features = features["labels"]
        label_features =  features["labels"] #[{"input_ids": feature["labels"]} for feature in features]

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


#
# class DataCollatorSpeechSeq2SeqWithPadding:
#     processor: Any
#     decoder_start_token_id: int
#
#     def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
#         # split inputs and labels since they have to be of different lengths and need different padding methods
#         # first treat the audio inputs by simply returning torch tensors
#         # import pdb; pdb.set_trace()
#         # input_features = features["input_features"]
#         input_features = [{"input_features": feature["input_features"]} for feature in features]
#         # import pdb;
#         # pdb.set_trace()
#         batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
#
#         # get the tokenized label sequences
#
#         # label_features = features["labels"]
#         label_features = [{"input_ids": feature["labels"]} for feature in features]
#         # pad the labels to max length
#         labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
#
#         # replace padding with -100 to ignore loss correctly
#         labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
#
#         # if bos token is appended in previous tokenization step,
#         # cut bos token here as it's append later anyways
#         if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
#             labels = labels[:, 1:]
#
#         batch["labels"] = labels
#
#         return batch



def evaluate(dataset, index, variant):
    data_point = dataset[index]
    sample = data_point[dataset.AUDIO_KEY]["array"]
    actual = normalize(variant.transcribe(audio=sample, language=dataset.LANGUAGE))
    target = normalize(data_point[dataset.TRANSCRIPTION_KEY])
    metrics = process_words(target, actual)
    return (actual, target, metrics)

def normalize(text):
    result = text.strip().lower()
    result = re.sub(r"[!\?\.,;]", "", result)
    return result

def evaluluate_model(model,dataset_iterable, tokenizer):


    actual_list = []
    target_list = []
    # List to store results
    results = []

    # Main evaluation loop
    for step, inputs in enumerate(dataloader):
        # Prediction step
        losses, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    for index in range(len(dataset)):
        actual, target, metrics = evaluate(dataset, index, variant)

        print("{0} / {1} {2}".format(index + 1, len(dataset), "-" * 70))
        print(actual)
        print(target)
        actual_list.append(actual)
        target_list.append(target)

        # Save the result as a dictionary in the list
        result = {
            "actual": actual,
            "target": target,
            "WER": metrics.wer
        }
        results.append(result)

        print("WER: {:2.1%}".format(metrics.wer))

    # Save the list as a JSON file
    with open("results/"+args.variant + "_" + args.dataset + ".json", "w") as outfile:
        json.dump(results, outfile, indent=4)

    combined_metrics = process_words(
        " ".join(target_list),
        " ".join(actual_list),
    )
    print(
        "Average WER of {0:2.1%} for {1} data points".format(
            combined_metrics.wer, len(dataset)
        )
    )


def steps_per_epoch(len_train_set,batch_size):

    if len_train_set % batch_size == 0:
        return int(len_train_set / batch_size)

    else:
       return int(len_train_set / batch_size) + 1


# fix bug when using default TuneBOHB searchers
# from ray.tune.search.bohb import TuneBOHB
#
# class TuneBOHB_fix(TuneBOHB):
#     @staticmethod
#     def convert_search_space(spec: Dict) -> "ConfigSpace.ConfigurationSpace":
#         resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)
#
#         if grid_vars:
#             raise ValueError(
#                 "Grid search parameters cannot be automatically converted "
#                 "to a TuneBOHB search space."
#             )
#
#         # Flatten and resolve again after checking for grid search.
#         spec = flatten_dict(spec, prevent_delimiter=True)
#         resolved_vars, domain_vars, grid_vars = parse_spec_vars(spec)
#
#         def resolve_value(
#             par: str, domain: Domain
#         ) -> ConfigSpace.hyperparameters.Hyperparameter:
#             quantize = None
#
#             sampler = domain.get_sampler()
#             if isinstance(sampler, Quantized):
#                 quantize = sampler.q
#                 sampler = sampler.sampler
#
#             if isinstance(domain, Float):
#                 if isinstance(sampler, LogUniform):
#                     lower = domain.lower
#                     upper = domain.upper
#                     if quantize:
#                         lower = math.ceil(domain.lower / quantize) * quantize
#                         upper = math.floor(domain.upper / quantize) * quantize
#                     return ConfigSpace.UniformFloatHyperparameter(
#                         par, lower=lower, upper=upper, log=True
#                     )
#                 elif isinstance(sampler, Uniform):
#                     lower = domain.lower
#                     upper = domain.upper
#                     if quantize:
#                         lower = math.ceil(domain.lower / quantize) * quantize
#                         upper = math.floor(domain.upper / quantize) * quantize
#                     return ConfigSpace.UniformFloatHyperparameter(
#                         par, lower=lower, upper=upper, log=False
#                     )
#                 elif isinstance(sampler, Normal):
#                     return ConfigSpace.hyperparameters.NormalFloatHyperparameter(
#                         par, mu=sampler.mean, sigma=sampler.sd, log=False
#                     )
#
#             elif isinstance(domain, Integer):
#                 if isinstance(sampler, LogUniform):
#                     lower = domain.lower
#                     upper = domain.upper
#                     if quantize:
#                         lower = math.ceil(domain.lower / quantize) * quantize
#                         upper = math.floor(domain.upper / quantize) * quantize
#                     else:
#                         # Tune search space integers are exclusive
#                         upper -= 1
#                     return ConfigSpace.UniformIntegerHyperparameter(
#                         par, lower=lower, upper=upper, log=True
#                     )
#                 elif isinstance(sampler, Uniform):
#                     lower = domain.lower
#                     upper = domain.upper
#                     if quantize:
#                         lower = math.ceil(domain.lower / quantize) * quantize
#                         upper = math.floor(domain.upper / quantize) * quantize
#                     else:
#                         # Tune search space integers are exclusive
#                         upper -= 1
#                     return ConfigSpace.UniformIntegerHyperparameter(
#                         par, lower=lower, upper=upper, log=False
#                     )
#
#             elif isinstance(domain, Categorical):
#                 if isinstance(sampler, Uniform):
#                     return ConfigSpace.CategoricalHyperparameter(
#                         par, choices=domain.categories
#                     )
#
#             raise ValueError(
#                 "TuneBOHB does not support parameters of type "
#                 "`{}` with samplers of type `{}`".format(
#                     type(domain).__name__, type(domain.sampler).__name__
#                 )
#             )
#
#         cs = ConfigSpace.ConfigurationSpace()
#         for path, domain in domain_vars:
#             par = "/".join(str(p) for p in path)
#             value = resolve_value(par, domain)
#             cs.add_hyperparameter(value)
#
#         return cs
