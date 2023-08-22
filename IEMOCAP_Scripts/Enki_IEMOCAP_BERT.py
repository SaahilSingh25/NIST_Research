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
metric = evaluate.load("accuracy")
model_name =  "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels = 3)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
cardiff_sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
seethal_sentiment_mapping = {"LABEL_0" : 0, "LABEL_1" : 1, "LABEL_2" : 2}

class IEMOCAP_PICKLE:
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)
        self.emotion_counts = defaultdict(Counter)
        self.utterances = []
        self.combined_ses = Counter()
        for script in self.data:
            self.emotion_counts[script["session"]][script["emotion"]] += 1
            self.utterances.append(script["transcription"])
            self.combined_ses.update(self.emotion_counts[script["session"]])

    def get_data(self):
        return self.data

    def get_emotion_counts(self):
        return self.emotion_counts

    def get_utterances(self):
        return self.utterances

    def get_combined_ses(self):
        return self.combined_ses

def determine_sentiment(data, classifier, label2id, filename, save_path):
    positive = ["hap", "exc"]
    negative = ["ang", "sad"]
    rows = []

    for i, row in enumerate(data.get_data(), 0):
        cur_emotion = row["emotion"]
        entry = [row["id"], row["transcription"], row["emotion"]]
        if cur_emotion in positive:
            entry.append(2)
        elif cur_emotion in negative:
            entry.append(0)
        else:
            entry.append(1)
        sentiment = label2id[(classifier(row["transcription"])[0])['label']]
        entry.append(sentiment)
        rows.append(entry)

    df = pd.DataFrame(rows, columns=["ID", "Utterance", "Emotion", "Sentiment", "Model Generated Sentiment"])
    df.to_csv(save_path + filename + '.csv', index=False)

def main():
    pickle_file = '../Dataset/IEMOCAP/data_collected_ang_exc_hap_neu_sad_signal_left_right_session_july24.pickle'
    csv_save_path = '../'

    enki_pickle = IEMOCAP_PICKLE(pickle_file)
    enki_utterances = enki_pickle.get_utterances()
    enki_data = enki_pickle.get_data()

    # seethal_classifier = pipeline("sentiment-analysis", model="Seethal/sentiment_analysis_generic_dataset")
    cardiff_classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
    determine_sentiment(enki_pickle, cardiff_classifier, cardiff_sentiment_mapping, "Cardiffnlp_Pretrained_Sentiment", csv_save_path)

if __name__ == "__main__":
    main()

