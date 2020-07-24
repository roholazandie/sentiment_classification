from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TextClassificationPipeline

tokenizer = DistilBertTokenizer.from_pretrained("./model_out/")
model = DistilBertForSequenceClassification.from_pretrained("./model_out/")

sentiment_classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)

result = sentiment_classifier("this is so cute!")
print(result)