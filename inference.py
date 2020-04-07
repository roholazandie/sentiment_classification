from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline
import numpy as np
import torch
import time

class SentimentClassifer():

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def __call__(self, text):
        input_ids = torch.tensor(self.tokenizer.encode(text, add_special_tokens=True)).unsqueeze(0)
        outputs = self.model(input_ids)
        outputs = outputs[0].detach().numpy()
        scores = np.exp(outputs) / np.exp(outputs).sum(-1)
        scores = scores[0].tolist()
        result = {"negative": scores[0], "neutral": scores[1], "positive": scores[2]}
        return result


def full_inference():
    '''
    full inference shows the case where we have the distribution of all sentiments
    :return:
    '''
    tokenizer = DistilBertTokenizer.from_pretrained("./model_out/")
    model = DistilBertForSequenceClassification.from_pretrained("./model_out/")

    sentiment_classifer = SentimentClassifer(model, tokenizer)

    result = sentiment_classifer("this is so cute!")
    print(result)


def simple_inference():
    '''
    this one is simpler and better for general case. It doesn't show the distribution of all the sentiments.
    this one uses the TextClassificationPipeline from transformers lib which is preferable
    :return:
    '''
    tokenizer = DistilBertTokenizer.from_pretrained("./model_out/")
    model = DistilBertForSequenceClassification.from_pretrained("./model_out/")
    model.to('cpu')
    sentiment_classifier = TextClassificationPipeline(model, tokenizer, device=-1)


    t1 = time.time()
    result = sentiment_classifier("this is so cute!")
    t2 = time.time()
    print(t2-t1, result)

    result = sentiment_classifier("That's so disgusting!")
    t3 = time.time()
    print(t3-t2, result)

    result = sentiment_classifier("this is a simple test.")
    t4 = time.time()
    print(t4-t3, result)


if __name__ == "__main__":
    full_inference()
    simple_inference()
