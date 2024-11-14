# standard
import os
import pprint
import pdb

from utils import  load_and_prepare_data_for_eval, DataCollatorSpeechSeq2SeqWithPadding, save_file, steps_per_epoch
import evaluate
import safetensors

import torch

# laod models
from transformers import WhisperFeatureExtractor
from transformers import WhisperTokenizer
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration
from transformers import set_seed

from transformers.models.whisper.convert_openai_to_hf import make_linear_from_emb

# For code organization and reporting
import configargparse
import logging

# For hyperparameter optimization
import ray
import ray.train

logger = logging.getLogger(__name__)


# We define all the different parameters for the training, model, evaluation etc.
# Whisper model type choices: https://huggingface.co/models?search=openai/whisper
# openai/whisper-large-v3 sofa

# https://stackoverflow.com/questions/37987839/how-can-i-run-tensorboard-on-a-remote-server
TUNE_CHOICES = ['small_small', 'large_small_BOHB', 'large_small_OPTUNA', 'large_large', '']
def parse_args():
    """ Parses command line arguments for the training """
    parser = configargparse.ArgumentParser()

    # Plotting

    # Training settings for Seq2SeqTrainingArguments
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Batch size per device")
    parser.add_argument("--output_tag", type=str,
                        default="whisper-tiny-de",
                        help="Base directory where model is save.")

    # model settings
    parser.add_argument("--model_type", type=str, default="openai/whisper-tiny", help="Model to optimize")
    parser.add_argument("--target_language", type=str, default="german", help="Target Language")
    # parser.add_argument("--load_model", action="store_true", help="Load model from model_ckpt_path")   # TODO: enable restoring: https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.restore.html#ray.tune.Tuner.restore
    parser.add_argument("--model_ckpt_path", type=str, default="", help="loads model from checkpoint training path")

    parser.add_argument("--search_schedule_mode", type=str, default="", choices=TUNE_CHOICES,
                        help="Which Searcher Algorithm and Scheduler combination. See 'get_searcher_and_scheduler' function for details.")

    # parser.add_argument("--device", type=str, default="cpu",
    #                     help="Path to audio batch-prepared audio files.")

    # Dataset settings
    parser.add_argument("--test_split", type=float, default=0.2, help="Percentage of test data.")
    parser.add_argument("--path_to_data", type=str, default="../data/datasets/fzh-wde0459_03_03", help="Path to audio batch-prepared audio files.")

    parser.add_argument("--fp16", action="store_true", default=False, help="Training with floating point 16 ")

    # Other settings
    parser.add_argument("--output_dir", type=str,
                        default="./output",
                        help="Base directory where outputs are saved.")
    # parser.add_argument("--tag", type=str, default="post_vacation",help="Tag added to save folder")
    # parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Debug mode (more log output, additional callbacks)")
    parser.add_argument("--random_seed", type=int, default=1337, help="Random Seed for reproducibility")
    parser.add_argument("--push_to_hub", action="store_true", help="Push best model to Hugging Face")
    parser.add_argument("-c", is_config_file=True, type=str, help="Config file path")

    args = parser.parse_args()
    return args

# Function to select the device
def select_device():
    # Check for CUDA availability with MPS support
    if torch.cuda.is_available():
        # Check if MPS is enabled and supported
        if torch.backends.mps.is_available():
            logger.info("Using CUDA MPS device")
            return torch.device("cuda")
        else:
            logger.info("Using standard CUDA device")
            return torch.device("cuda")
    else:
        logger.info("CUDA not available, using CPU")
        return torch.device("cpu")

def get_models(model_type,target_language):
    feature_extractor = WhisperFeatureExtractor.from_pretrained(model_type)
    tokenizer = WhisperTokenizer.from_pretrained(model_type, language=target_language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_type, language=target_language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(model_type)
    model.generation_config.language = target_language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    return model, feature_extractor, tokenizer, processor

def eval_model(args, eval_dict, data_collator=None):

    device = select_device()
    # logger.info('Device %s detected.', device)

    # get models
    model, feature_extractor, tokenizer, processor = get_models(args.model_type,args.target_language)


    # Load the state dictionary from the checkpoint
    if len(args.model_ckpt_path)>0:
        # checkpoint = torch.load(args.checkpoint_path)
        # model.load_state_dict(checkpoint["model_state_dict"])  # adjust if your checkpoint has a different key
        state_dict = safetensors.torch.load_file(os.path.join(args.model_ckpt_path, 'model.safetensors'))

        # https://github.com/openai/whisper/discussions/2302
        model.load_state_dict(state_dict, strict=False)
        model.proj_out = make_linear_from_emb(model.model.decoder.embed_tokens)

        # pdb.set_trace()
        # model = WhisperForConditionalGeneration.from_pretrained(args.model_ckpt_path, safe=True)
        logger.info('Whisper model from checkpoint %s loaded.', args.model_ckpt_path)
    else:
        logger.info('Whisper model %s loaded.', args.model_type)

    # Define metric for evaluation
    metric = evaluate.load("wer")
    def compute_metrics(pred_ids,label_ids):
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, "original": label_str, "predictions":pred_str}


    model.eval().to(device)        # eval mode for model pushed to device
    # Increase batch size if cuda is avaiable
    batch_size =  args.eval_batch_size
            #2*args.eval_batch_size) if device.type == "cuda" else args.eval_batch_size  # Larger batch size if CUDA is available

    # Prepare Test Set
    test_ds = ray.data.from_huggingface(eval_dict["test"])
    test_ds_iterable = test_ds.iter_torch_batches(
        batch_size=batch_size, collate_fn=data_collator
    )
    eval_results = {}
    count = -1
    wer_average = 0
    for batch in test_ds_iterable:
        count += 1
        # inputs, labels = batch["input_features"], batch["labels"]
        label_ids = batch["labels"] #tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        pred_ids = model.generate(batch["input_features"].to(device))

        outputs = compute_metrics(pred_ids,label_ids)

        eval_results[str(count)] = outputs
        wer_average += outputs["wer"]
        if count % 50:
            logger.info('WER for step %s...',count)
            logger.info('...%s',outputs["wer"])

    logger.info("WER average on Test Set %s", wer_average/(count+1))

    save_file(eval_results, args.output_dir, mode = 'json', file_tag='eval')

    # pdb.set_trace()




if __name__ == "__main__":
    # Arguments and Logger
    args = parse_args()
    logging.basicConfig(format="%(asctime)-5.5s %(name)-20.20s %(levelname)-7.7s %(message)s", datefmt="%H:%M", level=logging.DEBUG if args.debug else logging.INFO)
    logger.info("Hi!")

    # set random seed for reproducibility
    set_seed(args.random_seed)

    # get models
    model, feature_extractor, tokenizer, processor = get_models(args.model_type,args.target_language)

    args.path_to_data = r"../data/datasets"
    eval_dict, len_eval_set = load_and_prepare_data_for_eval(args.path_to_data, feature_extractor, tokenizer,
                                                          test_size=args.test_split, seed=args.random_seed)

    # if args.debug:
    #     args.warmup_steps = 0
    #     args.eval_steps = 1
    #     args.save_steps = 1
    #     args.max_steps = 2
    #     args.logging_steps = 1
    #     ray.init()
    #     pprint.pprint(ray.cluster_resources())
    #     args.path_to_data = r"../data/datasets"
    #     dataset_dict = load_and_prepare_data_from_folders(args.path_to_data, feature_extractor, tokenizer,
    #                                                       test_size=args.test_split, seed=args.random_seed)
    #     #         # args.pat_to_data = r"../data/datasets/fzh-wde0459_03_03"
    #     # dataset_dict = load_and_prepare_data_from_folder(args.path_to_data, feature_extractor, tokenizer,
    #     #                                                  test_size=args.test_split)
    # else:
    #     ray.init()
    #
    #     pprint.pprint(ray.cluster_resources())
    #     if args.path_to_data[-1] == 's':
    #         args.path_to_data = r"../data/datasets"  # /fzh-wde0459_03_03
    #         dataset_dict, len_train_set = load_and_prepare_data_from_folders(args.path_to_data, feature_extractor, tokenizer,
    #                                                      test_size=args.test_split)
    #     else:
    #         args.path_to_data = r"../data/datasets/fzh-wde0459_03_03"
    #         dataset_dict, len_train_set = load_and_prepare_data_from_folder(args.path_to_data, feature_extractor, tokenizer,
    #                                                       test_size=args.test_split) #, seed=args.random_seed)
    #
    #     logger.debug("Starting main_finetuning.py with arguments %s", args)
    #
    logger.info('len_eval_set: %s',len_eval_set)



    # save settings for reproducibility, TODO: add random seed
    config_ = 'Parsed args:\n{}\n\n'.format(pprint.pformat(args.__dict__))
    args.output_dir = os.path.join(args.output_dir, args.search_schedule_mode, args.output_tag )
    # pdb.set_trace()
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_file(config_,args.output_dir, file_tag = 'eval')

    # get the relevant hyperparameter config and training kwargs (necessary for finetuning with ray)
    # training_kwargs = make_training_kwargs(args)

    # prepare dataset dict
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # pdb.set_trace()
    eval_model(args, eval_dict, data_collator=data_collator)

