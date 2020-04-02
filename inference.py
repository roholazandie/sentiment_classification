from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline

tokenizer = DistilBertTokenizer.from_pretrained("./model_out/")
model = DistilBertForSequenceClassification.from_pretrained("./model_out/")

sentiment_classifier = TextClassificationPipeline(model, tokenizer)

result = sentiment_classifier("this is so cute!")
print(result)

result = sentiment_classifier("That's so disgusting!")
print(result)

result = sentiment_classifier("this is a simple test.")
print(result)