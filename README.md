# asr-christiian
Working Repo von Christian

# Week 1: 08.10 - 15.10

* [x] Proof of concept: get finetuning script from Peter working
      => improves performance on mini train set
* [x] prepare code for hyperparameter optimization
      => not seemless integration with ray tune (industry standard for hyperparameter optimization)
* [x] prepare presentation on hyperparameter optimization


# Week 2&3: 16.10 - 31.11

* [x] Set up High-Perfomance cluster
* [x] Submit evaluation job for baseline whisper model
* [x] prepare code for hyperparameter optimization (from Week 1)
=> Collator Function had to be adapted as Ray Tune datasets created a numpy object which could not be processed from HF tokenizer.pad class 
      
Next steps:
* [ ] Evaluate Baseline model (WhisperX e.g.) on test set to directly compare performance with fine-tuned model
* [x] Define hyperparameters to optimize
* [x] Run first dummy-finetuning on cluster


# WER-Results on test-set

| Model | Original | BOHB_jan |
|----------|----------|----------|
| tiny    | Data 1   | Data 2   |
| small    | 63.23   | Data 4   |
| medium   | 60.62   | Data 6   |

