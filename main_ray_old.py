# standard
import os
from functools import partial
import pprint
import pdb
import numpy as np
from time import strftime

from utils import  load_and_prepare_data_from_folder, load_and_prepare_data_from_folders, DataCollatorSpeechSeq2SeqWithPadding, save_file
import evaluate

# laod models
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import set_seed

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
from ray.train.torch import TorchTrainer
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
import ray.train
from ray.train.huggingface.transformers import prepare_trainer, RayTrainReportCallback
from ray.tune import Tuner
from ray.train import Checkpoint
# for training settings
import torch

logger = logging.getLogger(__name__)


# We define all the different parameters for the training, model, evaluation etc.
# Whisper model type choices: https://huggingface.co/models?search=openai/whisper
# openai/whisper-large-v3 sofa


TUNE_CHOICES = ['small_small', 'large_small_BOHB', 'large_small_OPTUNA', 'large_large']
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
    # parser.add_argument("--warmup_steps", type=int, default=500, help="Warumup gradient steps")
    # parser.add_argument("--lr", type=float, default=1e-5, help="Initial learning rate.")
    parser.add_argument("--generation_max_length", type=int, default=225, help="Max length of token output")
    parser.add_argument("--save_steps", type=int, default=1000, help="After how many steps to save the model?")
    parser.add_argument("--eval_steps", type=int, default=1000, help="After how many steps to evaluate model")
    parser.add_argument("--logging_steps", type=int, default=25, help="After how many steps to do some logging")


    # model settings
    parser.add_argument("--model_type", type=str, default="openai/whisper-tiny", help="Model to optimize")
    parser.add_argument("--target_language", type=str, default="german", help="Target Language")
    # parser.add_argument("--load_model", action="store_true", help="Load model from model_ckpt_path")   # TODO: enable restoring: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.restore.html#ray.tune.Tuner.restore
    # parser.add_argument("--model_ckpt_path", type=str, defult="", help="loads model from checkpoint training path")

    # Dataset settings
    parser.add_argument("--test_split", type=float, default=0.2, help="Percentage of test data.")
    parser.add_argument("--path_to_data", type=str, default="../data/datasets/fzh-wde0459_03_03", help="Path to audio batch-prepared audio files.")


    # Hyperparameter Optimization settings for Ray Tune
    # parser.add_argument("--tune", action="store_true", help="Do Hyperparameter optimization with Ray Tune")
    parser.add_argument("--num_samples", type=int, default=5, help="Total number of Ray Actors/ different hyperparameters configs to compare.")
    # parser.add_argument("--max_t", type=int, default=10, help="Max number of steps for finding the best hyperparameters")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of trials that can run at the same time")
    parser.add_argument("--cpus_per_trial", type=int, default=1, help="Number of CPUs per Ray actor")
    parser.add_argument("--gpus_per_trial", type=int, default=0, help="Number of GPUs per Ray actor")
    parser.add_argument("--use_gpu", action="store_true", help="If using GPU for the finetuning")
    # For Scheduler and Search algorithm specifically
    parser.add_argument("--search_schedule_mode", type=str, default="large_small_BOHB", choices=TUNE_CHOICES, help="Which Searcher Algorithm and Scheduler combination. See 'get_searcher_and_scheduler' function for details.")
    parser.add_argument("--reduction_factor", type=int, default=2, help="Factor by which trials are reduced after grace_period of epochs")
    parser.add_argument("--grace_period", type=int, default=1, help="With grace_period=n you can force ASHA to train each trial at least for n epochs.s")
    # For Population Based Training (PBT): https://docs.ray.io/en/latest/tune/api/doc/ray.tune.schedulers.PopulationBasedTraining.html
    parser.add_argument("--perturbation_interval", type=int, default=10, help="Models will be considered for perturbation at this interval of time_attr.")
    parser.add_argument("--burn_in_period", type=int, default=1, help="Grace Period for PBT")

    # Other settings
    parser.add_argument("--output_dir", type=str,
                        default="./output",
                        help="Base directory where outputs are saved.")
    # parser.add_argument("--tag", type=str, default="post_vacation",help="Tag added to save folder")
    # parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--random_seed", type=int, default=1337, help="Random Seed for reproducibility")
    parser.add_argument("--push_to_hub", action="store_true", help="Push best model to Hugging Face")
    parser.add_argument("--calculate_baseline", action="store_true", help="Calculates Performance (WER) on Test Set of original model (i.e. without finetuning)")
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")

    args = parser.parse_args()
    return args


# only those hyperparameter which should be optimized
def make_training_kwargs(args):
    # setting for seq2seq trainer
    #https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments
    # we are orientating ourselves on the orginal whisper hyperparameters https://cdn.openai.com/papers/whisper.pdf
    # this requires changing the beta1 and beta 2 hyperparameters of the AdamW optimizer

    training_kwargs = {
        "output_dir": args.output_dir,
        # "per_device_train_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        # "warmup_steps": args.warmup_steps,
        "max_steps": args.max_steps,
        "generation_max_length": args.generation_max_length,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps,
        "logging_steps": args.logging_steps,
        "eval_delay": int(args.max_steps/10),
        "adam_beta1": 0.9,
        "adam_beta2": 0.98,
        "seed": args.random_seed
    }
    # if args.tune:
    #     hyper_kwargs = {
    #         "learning_rate": tune.loguniform(args.lr/10,args.lr),  # sampling loguniformly a lr for each particle
    #     }
    # else:
    #     hyper_kwargs = {
    #         "learning_rate": args.lr,
    #     }

    return training_kwargs


def get_models(model_type,target_language):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
    tokenizer = WhisperTokenizer.from_pretrained(model_type, language=target_language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_type, language=target_language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_type)
    model.generation_config.language = target_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return model, feature_extractor, tokenizer, processor

def train_model(config, training_kwargs=None, data_collator=None):

    # get models
    model, feature_extractor, tokenizer, processor = get_models(args.model_type,args.target_language)

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

        # semantic_results = {}
        # semantic_results["original"] = label_str
        # semantic_results["pred_str"] = pred_str
        # save_semantic_path = os.path.join(training_kwargs["output_dir"], 'eval_results')
        # if not os.path.exists(save_semantic_path):
        #     os.makedirs(save_semantic_path)
        # time_ = strftime("%d_%m_%Y_%H_%M_%S")
        # np.save(os.path.join(save_semantic_path, time_ + '.npy'), semantic_results)

        return {"wer": wer}

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

    train_ds = ray.train.get_dataset_shard("train")
    eval_ds = ray.train.get_dataset_shard("eval")
    test_ds = ray.train.get_dataset_shard("test")

    # this is a hack - as train_ds from Ray requires the data_collotor, so does Seq2SeqTrainer from HF
    # but collating twice does not make sense, therefore we introduce the indentity collator
    def data_collator_id(input):
        return input

    # the data collator takes our pre-processed data and prepares PyTorch tensors ready for the model.
    train_ds_iterable = train_ds.iter_torch_batches(
        batch_size=config["per_device_train_batch_size"], collate_fn=data_collator
    )
    eval_ds_iterable = eval_ds.iter_torch_batches(
        batch_size=config["per_device_train_batch_size"], collate_fn=data_collator
    )
    test_ds_iterable = test_ds.iter_torch_batches(
        batch_size=config["per_device_train_batch_size"], collate_fn=data_collator
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds_iterable,
        eval_dataset=eval_ds_iterable,
        data_collator=data_collator_id,
        compute_metrics=compute_metrics,
        # we are orientating ourselves on the orginal whisper hyperparameters https://cdn.openai.com/papers/whisper.pdf
        # this requires changing the beta1 and beta 2 hyperparameters of the AdamW optimizer
        # optimizers = (torch.optim.AdamW())
        # tokenizer=tokenizer, TODO: make sure its not necessary since we do pre-processing already
    )



    trainer.add_callback(RayTrainReportCallback())
    trainer = prepare_trainer(trainer)


    # calculate baseline performance on test set
    # if True:
    logger.info("Start Evaluation baseline model on Test Set")
        # pdb.set_trace()
    baseline_results = trainer.evaluate(eval_dataset = test_ds_iterable)

    save_file(baseline_results, training_args.output_dir,  mode='eval_results', file_tag="baseline_on_test")

        # from utils import evaluluate_model
        # evaluluate_model(model,test_ds_iterable,tokenizer)


    trainer.train()
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    # Arguments and Logger
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("Hi!")

    # set random seed for reproducibility
    set_seed(args.random_seed)
    # param_space = {"seed": tune.randint(0, 1000)},

    # get models
    model, feature_extractor, tokenizer, processor = get_models(args.model_type,args.target_language)

    if args.debug:
        args.warmup_steps = 0
        args.eval_steps = 1
        args.save_steps = 1
        args.max_steps = 2
        args.logging_steps = 1
        ray.init()
        pprint.pprint(ray.cluster_resources())
        args.path_to_data = r"../data/datasets"
        dataset_dict = load_and_prepare_data_from_folders(args.path_to_data, feature_extractor, tokenizer,
                                                          test_size=args.test_split, seed=args.random_seed)
        # args.path_to_data = r"../data/datasets/fzh-wde0459_03_03"
        # dataset_dict = load_and_prepare_data_from_folder(args.path_to_data, feature_extractor, tokenizer,
        #                                                  test_size=args.test_split)
    else:
        ray.init()

        # ray.init(address="auto", _redis_password = os.environ["redis_password"])
        # register_ray()
        pprint.pprint(ray.cluster_resources())
        args.path_to_data = r"/home/chrvt/data/datasets"  # /fzh-wde0459_03_03
        # dataset_dict = load_and_prepare_data_from_folder(args.path_to_data, feature_extractor, tokenizer,
        #                                                  test_size=args.test_split)
        dataset_dict = load_and_prepare_data_from_folders(args.path_to_data, feature_extractor, tokenizer,
                                                          test_size=args.test_split, seed=args.random_seed)

        logger.debug("Starting main_finetuning.py with arguments %s", args)


    # save settings for reproducibility, TODO: add random seed
    config_ = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__))
    args.output_dir = os.path.join(args.output_dir, args.search_schedule_mode, args.output_tag )
    # pdb.set_trace()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_file(config_,args.output_dir)

    # get the relevant hyperparameter config and training kwargs (necessary for finetuning with ray)
    training_kwargs = make_training_kwargs(args)



    # prepare dataset dict
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # do hyperparameter optimization way ray tune or simply normal training
    ray_datasets = {
        "train": ray.data.from_huggingface(dataset_dict["train"]),
        "validation": ray.data.from_huggingface(dataset_dict["validation"]),
        "test": ray.data.from_huggingface(dataset_dict["test"]),
    }

    resources_per_trial={"CPU": args.cpus_per_trial, "GPU": args.gpus_per_trial}

    trainer = TorchTrainer(
        partial(train_model, training_kwargs=training_kwargs,
                data_collator=data_collator),
        scaling_config=ScalingConfig(num_workers=args.num_workers, use_gpu=args.use_gpu, resources_per_worker = resources_per_trial),
        datasets={
            "train": ray_datasets["train"],
            "eval": ray_datasets["validation"],
            "test": ray_datasets["test"],
        },
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="eval_loss",
                checkpoint_score_order="min",
            ),
        ),
    )

    # https://docs.ray.io/en/latest/tune/faq.html#which-search-algorithm-scheduler-should-i-choose
    # What is the strategy for chossing the right Optimization Algorithm and Scheduler for Ray Tune?
    # " Our go-to solution is usually to use random search with ASHA for early stopping for smaller problems.
    # Use BOHB for larger problems with a small number of hyperparameters and
    # Population Based Training for larger problems with a large number of hyperparameters if a learning schedule is acceptable."


    from ray.tune.search.bohb import TuneBOHB
    from ray.tune.search.basic_variant import BasicVariantGenerator



    def get_searcher_and_scheduler(args):
        # A search algorithm searches the Hyperparameter space for good combinations.
        #   The easises way of searching: grid search which is the BasicVariantGenerator.
        #   More advanced search algorithms are based on Bayesian Optimization strategies.
        #   https://docs.ray.io/en/latest/tune/api/suggestion.html
        # A scheduler can early terminate bad trials, pause trials, clone trials, and alter hyperparameters of a running trial.
        #   https://docs.ray.io/en/latest/tune/api/schedulers.html#tune-scheduler-pbt
        # Check out the FAQ: https://docs.ray.io/en/latest/tune/faq.html#how-does-early-termination-e-g-hyperband-asha-work

        if args.search_schedule_mode == 'small_small':
            scheduler = ASHAScheduler(
                max_t=args.max_steps,
                reduction_factor = args.reduction_factor,
                grace_period = args.grace_period,
            )#[0],
            searcher = BasicVariantGenerator()
            # pdb.set_trace()
        elif args.search_schedule_mode == 'large_small_BOHB':
            from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
            # https://docs.ray.io/en/latest/tune/api/schedulers.html#tune-scheduler-bohb
            scheduler = HyperBandForBOHB(
                time_attr="training_iteration", # defines the "time" as training iterations (epochs in our case)
                max_t=args.max_steps,
                reduction_factor=args.reduction_factor,
                stop_last_trials=False,
            )
            bohb_search = TuneBOHB(
                # space=config_space,  # If you want to set the space manually
            )
            searcher = tune.search.ConcurrencyLimiter(bohb_search, max_concurrent=4)

        elif args.search_schedule_mode == 'large_small_OPTUNA':
            from ray.tune.search.optuna import OptunaSearch
            # https://docs.ray.io/en/latest/tune/api/suggestion.html#tune-optuna
            scheduler = ASHAScheduler(
                max_t=args.max_steps,
                reduction_factor=args.reduction_factor,
                grace_period=args.grace_period,
            )#[0]
            searcher = OptunaSearch()

        elif args.search_schedule_mode == 'large_large':
            from ray.tune.schedulers import PopulationBasedTraining
            # https://docs.ray.io/en/latest/tune/api/schedulers.html#tune-scheduler-pbt
            # https://docs.ray.io/en/latest/tune/examples/pbt_guide.html
            scheduler = PopulationBasedTraining(
                time_attr='training_iteration',   # defines the "time" as training iterations (epochs in our case)
                perturbation_interval=args.perturbation_interval,
                hyperparam_mutations={
                    "train_loop_config": {
                               "learning_rate": tune.loguniform(1e-5, 1e-1),
                               "per_device_train_batch_size": tune.choice([2, 4, 8, 16]), #, 32, 64, 128, 256]),
                               "weight_decay": tune.uniform(0.0, 0.2),
                    #tune.grid_search([2e-5]), # , 2e-4, 2e-3, 2e-2
                # "max_steps": tune_epochs,
                        }
                    # "For learning rates, we suggest using a loguniform distribution between 1e-5 and 1e-1: tune.loguniform(1e-5, 1e-1)."
                    # "learning_rate": tune.loguniform(1e-5, 1e-1)
                    # "alpha": tune.uniform(0.0, 1.0),
                }
            )
            searcher = BasicVariantGenerator() # default searcher

        return searcher, scheduler

    tune_searcher, tune_scheduler  = get_searcher_and_scheduler(args)

    # "iterations": 100,
    # "width": tune.uniform(0, 20),
    # "height": tune.uniform(-100, 100),
    # "activation": tune.choice(["relu", "tanh"]),

    # {
    #     "per_gpu_batch_size": (16, 64),
    #     "weight_decay": (0, 0.3),
    #     "learning_rate": (1e-5, 5e-5),
    #     "warmup_steps": (0, 500),
    #     "num_epochs": (2, 5)
    # }
#The training function to execute on each worker.
# This function can either take in zero arguments or a single Dict argument which is set by defining train_loop_config.
    tuner = Tuner(
        trainer,
        param_space={
            #
            # "seed": tune.randint(0, 1000),   16, #
            "train_loop_config": {
                "learning_rate": tune.loguniform(1e-5, 1e-1),
                "warmup_steps": 2, #tune.choice([0,1,2]),
                "per_device_train_batch_size": 16, #tune.choice([2, 4, 8, 16]), #, 32, 64, 128, 256]),
                "weight_decay": tune.uniform(0.0, 0.2),
                    #tune.grid_search([2e-5]), # , 2e-4, 2e-3, 2e-2
                # "max_steps": tune_epochs,
            }
        },
        tune_config=tune.TuneConfig(
            metric="eval_loss",
            mode="min",
            num_samples=args.num_samples,
            search_alg=tune_searcher,
            scheduler= tune_scheduler,

        ),
        run_config=RunConfig(
            name="tune_transformers",
            checkpoint_config=CheckpointConfig(
                num_to_keep=1,
                checkpoint_score_attribute="eval_loss",
                checkpoint_score_order="min",
            ),
        ),
    )
    tune_results = tuner.fit()

    tune_results.get_dataframe().sort_values("eval_loss")

    best_result = tune_results.get_best_result()

    print('Best Result', best_result)


    # save best result into folder
    best_result_list = {}
    best_result_list["metrics"] = best_result.metrics
    best_result_list["path"] = best_result.path
    best_result_list["checkpoint"] = best_result.checkpoint
    np.save(os.path.join(args.output_dir,'best_result.npy'), best_result_list)
    # load
    # read_dictionary = np.load(os.path.join(args.output_dir,'best_result.npy'), allow_pickle='TRUE').item()



    # pdb.set_trace()
    if args.push_to_hub:
        # from huggingface_hub import notebook_login
        # notebook_login()
        checkpoint: Checkpoint = best_result.checkpoint

        with (checkpoint.as_directory() as checkpoint_dir):
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint")
            model = WhisperForConditionalGeneration.from_pretrained(checkpoint_path)

        model.push_to_hub('whisper_finetuning')




# for model evaluation

# import torch
# from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
# from datasets import load_dataset
#
#
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
# torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
#
# model_id = "openai/whisper-large-v3"
#
# model = AutoModelForSpeechSeq2Seq.from_pretrained(
#     model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
# )
# model.to(device)
#
# processor = AutoProcessor.from_pretrained(model_id)
#
# pipe = pipeline(
#     "automatic-speech-recognition",
#     model=model,
#     tokenizer=processor.tokenizer,
#     feature_extractor=processor.feature_extractor,
#     torch_dtype=torch_dtype,
#     device=device,
# )
#
# dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
# sample = dataset[0]["audio"]
#
# result = pipe(sample)
# print(result["text"])
