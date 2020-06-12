# Sentiment classification using transformers
This repository contains the code for sentiment classification for 3 class sentiment classification
which are {"negative", "neutral", "positive"}.

This is implemented with [transformers](https://github.com/huggingface/transformers) library from huggingface.

The current implementation uses distillbert but you can easily change the base model to anything else.

## Usage
You can download the pre-trained models [here](https://drive.google.com/open?id=1E4WJoqBe03vJxs8aSJJqNkXGpWrsPfRd). Extract and then use it like:

```python

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline

tokenizer = DistilBertTokenizer.from_pretrained("./model_out/")
model = DistilBertForSequenceClassification.from_pretrained("./model_out/")

sentiment_classifier = TextClassificationPipeline(model, tokenizer)

result = sentiment_classifier("this is so cute!")
print(result)

[{'label': 'positive', 'score': 0.9996757}]
```
### RESTful client app
You can also run this as a restful app. You just need to run app.py and then call it with:

```nashorn js
curl --header "Content-Type: application/json" --request POST --data '{"sentence":"You are so cute!"}' http://localhost:5555/api/rest/classify_sentiment
```
You can change the config for port and other settings in config.json.

## Train
The current dataset used here is from SST [stanford sentiment treebank](http://nlp.stanford.edu/~socherr/stanfordSentimentTreebank.zip).

Download the dataset, extract and then use the script create_dataset.py to create train.tsv, test.tsv and dev.tsv files.

```
python create_dataset.py --raw_dataset_dir <path_to_raw_dataset> --output_dir <path_to_output_dir>
```

After you convert the dataset to proper format. You can train:

```
python train.py --model_type distilbert --model_name_or_path distilbert-base-cased --task_name sst-full --do_train --do_eval --do_lower_case --data_dir <dataset_dir> --max_seq_length 128 --per_gpu_eval_batch_size=8 --per_gpu_train_batch_size=8 --learning_rate 2e-5 --num_train_epochs 10 --eval_all_checkpoints --output_dir <model_out_dir> --save_steps 2000 --logging_steps 500 --evaluate_during_training --overwrite_cache
```