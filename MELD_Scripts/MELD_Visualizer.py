import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv
import torch
from collections import Counter, defaultdict
from sklearn.metrics import roc_curve, auc, classification_report

class MELD_CSV:
    def __init__(self, csv_file):
        self.df = pd.read_csv(csv_file)
        self.emotion_counts = Counter(self.df['Emotion'])

    def get_emotion_counts(self):
        return self.emotion_counts

    def get_data(self):
        return self.df

    def get_utterances(self):
        return self.df['Utterance'].tolist()
    
    def get_sentiments(self):
        sentiment_mapping = {"negative": 0, "neutral": 1, "positive": 2}
        return self.df['Sentiment'].map(sentiment_mapping)

def visualize_emotion_counts(data, filename, save_path):
    sorted_data = sorted(data.items())
    keys = [item[0] for item in sorted_data]
    values = [item[1] for item in sorted_data]

    plt.figure(figsize=(10, 8))
    plt.bar(keys, values)
    plt.title(filename + ' Dataset', fontsize = 24)
    plt.xlabel('Categorized Emotion', fontsize = 22)
    plt.ylabel('Frequency', fontsize = 22)
    plt.xticks(fontsize = 18)
    plt.yticks(fontsize = 18)
    #plt.yticks(range(0, 501, 50))
    plt.savefig(save_path + filename + '_MELD_Emotion_Distribution.png')

def write_to_csv(data, filename, save_path):
    df = pd.DataFrame(sorted(list(data.items()),key= lambda x: x[0]), columns=['Emotion', 'Count'])
    df.to_csv(save_path + filename + '_MELD_Emotion_Distribution.csv', index=False)

def plot_roc(csv_file, save_path):
    df = pd.read_csv(csv_file)
    for model in ['Cardiffnlp Pretrained', 'Seethal Pretrained']:
        df[model + '_match'] = (df['Ground Truth'] == df[model]).astype(int)

    for i, model in enumerate(['Cardiffnlp Pretrained', 'Seethal Pretrained']):
        fpr, tpr, _ = roc_curve(df[model + '_match'], df['Ground Truth'])
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic for {}'.format(model))
        plt.legend(loc="lower right")
        if i == 0:
            plt.savefig('../MELD_ROC_Cardiffnlp.png')
        else:
            plt.savefig('../MELD_ROC_Seethal.png') 

def map_categorical_to_sentiment(file_path):
    df = pd.read_csv(file_path)
    pos = ['joy', 'surprise']
    neg = ['anger', 'disgust', 'fear', 'sadness']
    mapping = []
    
    for label in df['Emotion']:
        if label in pos:
            mapping.append('2')
        elif label in neg:
            mapping.append('0')
        else:
            mapping.append('1')

    df['Mapping_Sentiment'] = mapping
    df.to_csv(file_path, index=False)

def calculate_metrics(file_path):
    df = pd.read_csv(file_path)

    columns_to_compare = ['Mapping_Sentiment', 'Pretrained_Cardiffnlp', 'Finetuned_Cardiffnlp']
    metrics = {}

    for column in columns_to_compare:
        report = classification_report(df['Ground Truth'], df[column], output_dict=True)
        metrics[column] = report['weighted avg']

    metrics_df = pd.DataFrame(metrics).transpose()
    metrics_df.reset_index(inplace=True)
    metrics_df.rename(columns={'index': 'Method'}, inplace=True)

    # Since 'Method' column is common in both dataframes, we need to set 'Method' as index before joining
    df.set_index('Method', inplace=True)
    metrics_df.set_index('Method', inplace=True)

    # Join the original dataframe with the metrics dataframe
    df = df.join(metrics_df, how='outer')

    # Reset the index and save the updated dataframe back to the csv file
    df.reset_index(inplace=True)
    df.to_csv(file_path, index=False)

    return metrics_df

def main():
    dev_file = '../Dataset/MELD/dev_sent_emo.csv'
    test_file = '../Dataset/MELD/test_sent_emo.csv'
    train_file = '../Dataset/MELD/train_sent_emo.csv'
    image_save_path = '../Generated_Files/Images/MELD/'
    csv_save_path = '../Generated_Files/MELD/'

    dev = MELD_CSV(dev_file)
    test = MELD_CSV(test_file)
    train = MELD_CSV(train_file)

    calculate_metrics('../Generated_Files/MELD/MELD_Various_Sentiments.csv')

    '''
    map_categorical_to_sentiment('../Generated_Files/MELD/Categorical_to_Sentiment.csv')
    # Visualize emotion counts
    dev_ec = dev.get_emotion_counts()
    test_ec = test.get_emotion_counts()
    train_ec = train.get_emotion_counts()

    # Combine emotion counts
    comb_ec = {}
    comb_ec.update(dev_ec)
    for key in test_ec:
        comb_ec[key] += test_ec[key] + train_ec[key]

    
    visualize_emotion_counts(dev_ec, "Dev", image_save_path)
    visualize_emotion_counts(test_ec, "Test", image_save_path)
    visualize_emotion_counts(train_ec, "Train", image_save_path)
    visualize_emotion_counts(comb_ec, "Combined", image_save_path)

    # Write emotion counts to CSV files
    write_to_csv(dev_ec, "Dev", csv_save_path)
    write_to_csv(test_ec, "Test", csv_save_path)
    write_to_csv(train_ec, "Train", csv_save_path)
    write_to_csv(comb_ec, "Combined", csv_save_path)
    
    plot_roc('../Models/MELD/MELD_Various_Sentiments.csv', '../')
    '''
if __name__ == "__main__":
    main()
