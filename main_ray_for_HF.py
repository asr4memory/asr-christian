# standard
import os
from functools import partial
import pprint
import pdb

from utils import  load_and_prepare_data_from_folder, DataCollatorSpeechSeq2SeqWithPadding
import evaluate

# laod models
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer

# For code organization and reporting
import configargparse
import logging

# For hyperparameter optimization
from ray import tune
from ray import train
from ray.tune.schedulers import ASHAScheduler
import ray.cloudpickle as pickle
from ray.train.huggingface.transformers import prepare_trainer
import ray

logger = logging.getLogger(__name__)

# We define all the different parameters for the training, model, evaluation etc.
def parse_args():
    """ Parses command line arguments for the training """
    parser = configargparse.ArgumentParser()

    # Plotting

    # Training settings for Seq2SeqTrainingArguments
    parser.add_argument("--per_device_train_batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="increase by 2x for every 2x decrease in batch size")
    parser.add_argument("--output_tag", type=str,
                        default="whisper-tiny-de",
                        help="Base directory where model is save.")
    parser.add_argument("--max_steps", type=int, default=1000, help="Max Number of gradient steps")
    parser.add_argument("--warmup_steps", type=int, default=500, help="Warumup gradient steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--generation_max_length", type=int, default=225, help="Max length of token output")
    parser.add_argument("--save_steps", type=int, default=1000, help="After how many steps to save the model?")
    parser.add_argument("--eval_steps", type=int, default=1000, help="After how many steps to evaluate model")
    parser.add_argument("--logging_steps", type=int, default=25, help="After how many steps to do some logging")

    # model settings
    parser.add_argument("--model_type", type=str, default="openai/whisper-tiny", help="Model to optimize")
    parser.add_argument("--target_language", type=str, default="German", help="Target Language")

    # Dataset settings
    parser.add_argument("--test_split", type=float, default=0.2, help="Percentage of test data.")
    parser.add_argument("--path_to_data", type=str, default="../data/datasets/fzh-wde0459_03_03", help="Path to audio batch-prepared audio files.")

    # Hyperparameter Optimization settings for Ray Tune
    parser.add_argument("--tune", action="store_true", help="Do Hyperparameter optimization with Ray Tune")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of Particles/Branches of model")
    parser.add_argument("--max_t", type=int, default=10, help="Max number of steps for finding the best hyperparameters")
    parser.add_argument("--cpus_per_trial", type=int, default=1, help="Number of CPUs per particle")
    parser.add_argument("--gpus_per_trial", type=int, default=0, help="Number of GPUs per particle")

    # Other settings
    parser.add_argument("--output_dir", type=str,
                        default="output",
                        help="Base directory where outputs are saved.")
    # parser.add_argument("--tag", type=str, default="post_vacation",help="Tag added to save folder")
    # parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")

    args = parser.parse_args()
    return args


# only those hyperparameter which should be optimized
def make_training_kwargs(args):
    training_kwargs = {
        "output_dir": os.path.join(args.output_dir, args.output_tag),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "generation_max_length": args.generation_max_length,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
    }
    if args.tune:
        hyper_kwargs = {
            "learning_rate": tune.loguniform(args.lr/10,args.lr),  # sampling loguniformly a lr for each particle
        }
    else:
        hyper_kwargs = {
            "learning_rate": args.lr,
        }

    return hyper_kwargs, training_kwargs

# Define metric for evaluation
metric = evaluate.load("wer")
def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def get_models(model_type,target_language):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
    tokenizer = WhisperTokenizer.from_pretrained(model_type, language=target_language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_type, language=target_language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_type)
    model.generation_config.language = target_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return model, feature_extractor, tokenizer, processor

### load models
# whisper_model = "openai/whisper-tiny"           #TODO: add to arguments
# target_language = "German"                      #TODO: add to arguments







# training_args = Seq2SeqTrainingArguments(
#     output_dir="whisper-tiny-de",  # change to a repo name of your choice
#     per_device_train_batch_size=16,
#     gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
#     learning_rate=1e-5,
#     warmup_steps=0,  #500
#     max_steps=10,    #1000
#     gradient_checkpointing=True,
#     fp16=False,
#     evaluation_strategy="steps",
#     per_device_eval_batch_size=8,
#     predict_with_generate=True,
#     generation_max_length=225,
#     save_steps=1000,
#     eval_steps=1, #1000
#     logging_steps=25,
#     report_to=["tensorboard"],
#     load_best_model_at_end=True,
#     metric_for_best_model="wer",
#     greater_is_better=False,
#     push_to_hub=False,
# )
#
# trainer = Seq2SeqTrainer(
#     args=training_args,
#     model=model,
#     train_dataset=dataset_dict["train"],
#     eval_dataset=dataset_dict["test"],
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     tokenizer=processor.feature_extractor,
# )
# processor.save_pretrained(training_args.output_dir)



def train_model(config, training_kwargs=None, model=None, dataset_dict=None, data_collator=None, compute_metrics=None, tokenizer=None, tune=False):

    training_args = Seq2SeqTrainingArguments(
        gradient_checkpointing=True,
        fp16=False,
        evaluation_strategy="steps",
        predict_with_generate=True,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        push_to_hub=False,
        **config,
        **training_kwargs,
    )

    if tune:
        ray_datasets = {
            "train": ray.data.from_huggingface(dataset_dict["train"]),
            "validation": ray.data.from_huggingface(dataset_dict["validation"]),
            "test": ray.data.from_huggingface(dataset_dict["test"]),
        }

        pdb.set_trace()
        train_ds = ray.train.get_dataset_shard("train")
        eval_ds = ray.train.get_dataset_shard("eval")

        train_ds_iterable = train_ds.iter_torch_batches(
            batch_size=16, collate_fn=collate_fn
        )
        eval_ds_iterable = eval_ds.iter_torch_batches(
            batch_size=16, collate_fn=collate_fn
        )

        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=train_ds_iterable,
            eval_dataset=eval_ds_iterable,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )
    else:
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=dataset_dict["train"],
            eval_dataset=dataset_dict["test"],
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
        )


    if tune:
        trainer.add_callback(RayTrainReportCallback())
        trainer = prepare_trainer(trainer)

    trainer.train()
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    # Arguments and Logger
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("Hi!")

    if args.debug:
        args.warmup_steps = 0
        args.eval_steps = 1
        args.save_steps = 10
        args.max_steps = 10
        args.logging_steps = 2
        ray.init()
        pprint(ray.cluster_resources())


    logger.debug("Starting main_finetuning.py with arguments %s", args)

    # save settings for reproducibility
    config = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__))
    config_path = os.path.join(args.output_dir, 'config'+ args.output_tag+'.txt')
    if not os.path.exists(config_path):
        # pdb.set_trace()
        # os.makedirs(os.path.join(os.getcwd(),args.output_dir))
        with open(config_path, 'a') as f:
            print(config, file=f)


    # get the relevant hyperparameter config and training kwargs (necessary for finetuning with ray)
    config, training_kwargs = make_training_kwargs(args)

    # get models
    model, feature_extractor, tokenizer, processor = get_models(args.model_type,args.target_language)

    # prepare dataset dict
    dataset_dict = load_and_prepare_data_from_folder(args.path_to_data, feature_extractor, tokenizer, test_size=args.test_split)
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # do hyperparameter optimization way ray tune or simply normal training
    if args.tune:
        # TODO: understand settings better
        # max_num_epochs = 10
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=args.max_t,
            grace_period=1,
            reduction_factor=2,
        )

        # gpus_per_trial = 1
        # num_samples = 5
        result = tune.run(
            partial(train_model, training_kwargs=training_kwargs, model=model, dataset_dict=dataset_dict, data_collator=data_collator, compute_metrics=compute_metrics, tokenizer=tokenizer),
            resources_per_trial={"cpu": args.cpus_per_trial, "gpu": args.gpus_per_trial},
            config=config,
            num_samples=args.num_samples,
            scheduler=scheduler,
        )
        best_trial = result.get_best_trial("loss", "min", "last")
    else:
        train_model(config, training_kwargs=training_kwargs, model=model, dataset_dict=dataset_dict, data_collator=data_collator, compute_metrics=compute_metrics, tokenizer=tokenizer)


    # main(num_samples=500, max_num_epochs=1000, gpus_per_trial=0)


# Gesamtes DatasetDict anzeigen
# print(dataset_dict)
#
# check if feature extractor and tokenizer works properly
# input_str = dataset_dict["train"][0]["transcription"]
# labels = tokenizer(input_str).input_ids
# decoded_with_special = tokenizer.decode(labels, skip_special_tokens=False)
# decoded_str = tokenizer.decode(labels, skip_special_tokens=True)

# print(f"Input:                 {input_str}")
# print(f"Decoded w/ special:    {decoded_with_special}")
# print(f"Decoded w/out special: {decoded_str}")
# print(f"Are equal:             {input_str == decoded_str}")