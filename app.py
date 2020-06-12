from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline
from flask import Flask, request
import json
from config import SentimentClassificationConfig

config = SentimentClassificationConfig.from_json("config.json")

app = Flask(__name__)

tokenizer = DistilBertTokenizer.from_pretrained(config.model_path)
model = DistilBertForSequenceClassification.from_pretrained(config.model_path)

if config.use_cuda:
    model = model.cuda()

@app.route('/api/rest/classify_sentiment', methods=["POST"])
def classify_sentiment():
    rest_request = json.loads(request.data.decode('utf-8'))
    sentence = str(rest_request["sentence"])
    sentiment_classifier = TextClassificationPipeline(model, tokenizer, device=0 if config.use_cuda else -1)

    result = sentiment_classifier(sentence)
    return str(result)

if __name__ == '__main__':
    app.run(host=config.host, port=config.port, debug=True)


#curl --header "Content-Type: application/json" --request POST --data '{"sentence":"You are so cute!"}' http://localhost:5555/api/rest/classify_sentiment