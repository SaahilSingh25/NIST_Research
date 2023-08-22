import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import csv
import torch
import evaluate
from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, pipeline
from collections import Counter, defaultdict
from torch.nn.functional import softmax
from torchinfo import summary
from datasets import load_metric, load_dataset, Dataset

global metric, data_collator, model_name, tokenizer, model, sentiment_mapping
''' metric = evaluate.load("accuracy")
model_name =  "../Models/MELD/Model_MELD_Cardiffnlp"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer) '''
sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
cardiff_sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
seethal_sentiment_mapping = {"LABEL_0" : 0, "LABEL_1" : 1, "LABEL_2" : 2}

class MELD_CSV:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.df = pd.DataFrame(self.df)
        self.df['label'] = self.df['Sentiment'].map(sentiment_mapping)
        self.emotion_counts = Counter(self.df['Emotion'])

    def get_emotion_counts(self):
        return self.emotion_counts

    def get_data(self):
        return self.df

    def get_preprocessing_data(self):
        data = []
        for index, row in self.get_data().iterrows():
            entry = {}
            entry["Utterance"] = row["Utterance"]
            entry["label"] = sentiment_mapping[row["Sentiment"]]
            data.append(entry)
        return data

    def get_utterances(self):
        return self.df['Utterance']

    def get_sentiments(self):
        return self.df['Sentiment']

def preprocess_function(element):
    encoded = tokenizer(element["Utterance"], truncation=True)
    return encoded

def compute_metrics(eval_pred):
   logits, labels = eval_pred
   predictions = np.argmax(logits, axis=-1)
   accuracy = metric.compute(predictions=predictions, references=labels)["accuracy"]
   return {"accuracy": accuracy}

def determine_sentiment(data, classifier, label2id, filename, save_path):
    positive = ["surprise", "joy"]
    negative = ["anger", "disgust", "fear", "sadness"]
    result = []

    for index, row in data.iterrows():
        if index == 20:
            break
        entry = [row["Sr No."], row["Utterance"], row["Emotion"]]
        cur_sent = row["Sentiment"]
        if cur_sent == "positive":
            entry.append(2)
        elif cur_sent == "negative":
            entry.append(0)
        else:
            entry.append(1)
        sentiment = label2id[(classifier(row["Utterance"])[0])['label']]
        entry.append(sentiment)
        result.append(entry)

    df = pd.DataFrame(result, columns=["ID", "Utterance", "Emotion", "Sentiment", "Model Generated Sentiment"])
    df.to_csv(save_path + filename + '.csv', index=False)

# Main function
def main():
    dev_file = '../Dataset/MELD/dev_sent_emo.csv'
    test_file = '../Dataset/MELD/test_sent_emo.csv'
    train_file = '../Dataset/MELD/train_sent_emo.csv'
    csv_save_path = '../'

    dev = MELD_CSV(dev_file)
    test = MELD_CSV(test_file)
    train = MELD_CSV(train_file)
    
    '''
    for i, param in enumerate(model.parameters(), 0):
        if i == (len(list(model.parameters())) - 2):
            break
        param.requires_grad = False

    batch_size = 128
    print(summary(model, input_size=(batch_size, 512), dtypes = [torch.long]))

    dev_dataset = Dataset.from_pandas(dev.get_data())
    train_dataset = Dataset.from_pandas(train.get_data())
    test_dataset = Dataset.from_pandas(test.get_data())

    tokenized_dev = dev_dataset.map(preprocess_function, batched=True)
    tokenized_train = train_dataset.map(preprocess_function, batched=True)
    tokenized_test = test_dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        # (9998 data points / 128 batch size ~ 78 steps)
        output_dir='/content/drive/MyDrive/NIST/Models/MELD/MELD_Seethal_Model_100_Epochs',
        optim='adamw_torch',
        num_train_epochs= 100,  # The total number of training epochs to perform.
        per_device_train_batch_size=128,  # The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size=16,  # The batch size per GPU/TPU core/CPU for evaluation.
        weight_decay=0.01,
        save_total_limit = 1,
        save_strategy = 'epoch',
        logging_strategy='epoch',
        load_best_model_at_end=True,
        evaluation_strategy='epoch',
        metric_for_best_model='accuracy',  # The metric to use to compare two different models.
        gradient_accumulation_steps=2,  # The number of steps to accumulate gradients before performing an optimizer step
        fp16=True  # 16-bit (mixed) precision training instead of 32-bit training
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_dev,
        compute_metrics=compute_metrics,
        data_collator=data_collator
        #callbacks=[EarlyStoppingCallback(early_stopping_patience=10)] # callbacks, early stopping if model stops improving
    ) 

    trainer.train()
    print(trainer.evaluate(eval_dataset=tokenized_dev))
    print(trainer.evaluate(eval_dataset=tokenized_test))   
    tokenizer.save_pretrained("/content/drive/MyDrive/NIST/Models/MELD/MELD_Seethal_Model_100_Epochs")
    trainer.save_model('/content/drive/MyDrive/NIST/Models/MELD/MELD_Seethal_Model_100_Epochs')
    '''

    finetuned_cardiffnlp_classifier = pipeline("sentiment-analysis", model="../Models/MELD/Model_MELD_Cardiffnlp")
    # pretrained_cardiffnlp_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    # finetuned_seethal_classifier = pipeline("sentiment-analysis", model="/content/drive/MyDrive/NIST/Models/MELD")
    # pretrained_seethal_classifier = pipeline("sentiment-analysis", model="Seethal/sentiment_analysis_generic_dataset")
    
    determine_sentiment(train.get_data(), finetuned_cardiffnlp_classifier, cardiff_sentiment_mapping, 'Cardiffnlp_FineTuned_Train_Sentiment', csv_save_path)

# Call main function if script is run directly
if __name__ == "__main__":
    main()
