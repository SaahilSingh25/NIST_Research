import pandas as pd
import pickle
import csv
import matplotlib.pyplot as plt
import chardet
import seaborn as sn
import numpy as np
from collections import Counter, defaultdict
from sklearn.metrics import roc_curve, auc
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

class IEMOCAP_PICKLE:
    def __init__(self, pickle_file):
        with open(pickle_file, 'rb') as f:
            self.data = pickle.load(f)

        self.emotion_counts = defaultdict(Counter)
        self.utterances = []
        self.combined_ses = Counter()
        self.categorical_to_valence = {'ang' : [], 'exc' : [], 'neu' : [], 'hap' : [], 'sad' : []}
        self.categorical_to_arousal = {'ang' : [], 'exc' : [], 'neu' : [], 'hap' : [], 'sad' : []}
        self.categorical_to_dominance = {'ang' : [], 'exc' : [], 'neu' : [], 'hap' : [], 'sad' : []}

        for script in self.data:
            cur_emotion = script['emotion']
            self.emotion_counts[script["session"]][cur_emotion] += 1
            self.utterances.append(script["transcription"])
            self.combined_ses.update(self.emotion_counts[script["session"]])
            self.categorical_to_valence[cur_emotion].append(script['v'])
            self.categorical_to_arousal[cur_emotion].append(script['a'])
            self.categorical_to_dominance[cur_emotion].append(script['d'])

    def get_data(self):
        return self.data

    def get_emotion_counts(self):
        return self.emotion_counts

    def get_utterances(self):
        return self.utterances

    def get_combined_ses(self):
        return self.combined_ses

    def get_categorical_to_dimensional(self):
        return [self.categorical_to_valence, self.categorical_to_arousal, self.categorical_to_dominance]

def visualize_emotion_counts(data, iave_path):
    combined_ses = Counter()
    for i, cur_ec in enumerate(data.values(), 1):
        sorted_cur_ec = sorted(cur_ec.items())
        keys = [item[0] for item in sorted_cur_ec]
        values = [item[1] for item in sorted_cur_ec]
        combined_ses.update(cur_ec)

        plt.figure(figsize=(10, 8))
        plt.bar(keys, values)
        plt.title(f'Session {i}', fontsize = 24)
        plt.xlabel('Categorized Emotion', fontsize = 22)
        plt.ylabel('Frequency', fontsize = 22)
        plt.xticks(fontsize = 18)
        plt.yticks(fontsize = 18)
        plt.yticks(range(0, 401, 50))
        plt.savefig(f'{save_path}Enki_IEMOCAP_Session{i}_Emotion_Distribution.png')

    sorted_combined_ses = sorted(combined_ses.items())
    combined_keys = [item[0] for item in sorted_combined_ses]
    combined_values = [item[1] for item in sorted_combined_ses]

    plt.figure(figsize=(10, 8))
    plt.bar(combined_keys, combined_values)
    plt.title('All Sessions', fontsize=24)
    plt.xlabel('Categorized Emotion', fontsize=22)
    plt.ylabel('Frequency', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(f'{save_path}All_Enki_IEMOCAP_Sessions_Emotion_Distribution.png')

    return combined_ses

def write_to_csv(data, save_path, is_combined):
    if not is_combined:
        for i, cur_ec in enumerate(data.values(), 1):
            df = pd.DataFrame(sorted(list(cur_ec.items()), key=lambda x: x[0]), columns=['Emotion', 'Frequency'])
            df.to_csv(f'{save_path}Enki_IEMOCAP_Session{i}_Emotion_Distribution.csv', index=False)
    else:
        df = pd.DataFrame(sorted(list(data.items()), key=lambda x: x[0]), columns=['Emotion', 'Frequency'])
        df.to_csv(f'{save_path}Enki_IEMOCAP_All_Sessions_Emotion_Distribution.csv', index=False)

def plot_roc(csv_file, save_path):
    df = pd.read_csv(csv_file)
    for model in ['Cardiffnlp Pretrained', 'Seethal Pretrained']:
        df[model + '_match'] = (df['Mapping'] == df[model]).astype(int)

    for i, model in enumerate(['Cardiffnlp Pretrained', 'Seethal Pretrained']):
        fpr, tpr, _ = roc_curve(df[model + '_match'], df['Mapping'])
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
            plt.savefig('../Enki_ROC_Cardiffnlp.png')
        else:
            plt.savefig('../Enki_ROC_Seethal.png')

def write_mapping_to_csv(data, save_path):
    res = []
    for script in data:
        utterance_id = script['id']
        utterance = script['transcription']
        emotion = script['emotion']
        valence = script['v']
        arousal = script['a']
        dominance = script['d']
        res.append([utterance_id, utterance, emotion, valence, arousal, dominance])

    df = pd.DataFrame(res, columns=['Utterance_ID', 'Utterance', 'Emotion', 'Valence', 'Arousal', 'Dominance'])
    df.to_csv(f'{save_path}utterance_emotion_dimensional.csv', index=False)

def k_means_cluster_analysis(k, file_path, save_path):
    df = pd.read_csv(file_path)
    km = KMeans(n_clusters=k, algorithm = 'elkan')
    vad_predicted = km.fit_predict(df[['Valence','Arousal','Dominance']])
    print(km.cluster_centers_)
    df['cluster'] = vad_predicted
    df.to_csv(file_path, index=False)


def apply_tsne_and_plot(file_path):
    df = pd.read_csv(file_path)
    features = df[['Valence', 'Arousal', 'Dominance']]
    #labels = df['Emotion'].replace('exc', 'hap')
    labels = df['Manual_Sentiment']
    df_sorted = df.sort_values('Manual_Sentiment', ascending=False)
    df_sorted.to_csv('Sorted_Mapping.csv', index=False)
    return
    
    tsne = TSNE(n_components=2, perplexity=30, early_exaggeration=12, learning_rate='auto', n_iter=1000, init='pca', method='barnes_hut')
    tsne_results = tsne.fit_transform(features)
    tsne_results = np.vstack((tsne_results.T, labels)).T
    tsne_df = pd.DataFrame(data=tsne_results, columns=('Dim_1', 'Dim_2', 'cluster_label'))

    markers = ['+', '*', 'd', '.', '>', '|']
    cluster_markers = {cluster: marker for cluster, marker in zip(tsne_df['cluster_label'].unique(), markers)}
    plt.figure(figsize=(10, 10))

    for label, marker in cluster_markers.items():
        plt.scatter(tsne_df.loc[tsne_df['cluster_label'] == label, 'Dim_1'], 
                    tsne_df.loc[tsne_df['cluster_label'] == label, 'Dim_2'], 
                    marker=marker, s=20,
                    label=label)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend() 
    plt.savefig('../All_Sessions_Manual_TSNE.png') 

def main():
    pickle_file = '../Dataset/IEMOCAP/data_collected_ang_exc_hap_neu_sad_signal_left_right_session_july24.pickle'
    cat_to_dim = '../Generated_Files/CSV_Files/Enki_IEMOCAP/categorical_to_dimensional.csv'
    image_save_path = '../Generated_Files/Images/Enki_IEMOCAP/'
    csv_save_path = '../Generated_Files/Enki_IEMOCAP'

    enki_pickle = IEMOCAP_PICKLE(pickle_file)
    apply_tsne_and_plot('../Generated_Files/Enki_IEMOCAP/Results/vad_clustering.csv')
    # k_means_cluster_analysis(3, cat_to_dim, image_save_path)
    # write_mapping_to_csv(enki_pickle.get_data(), csv_save_path)
    # enki_emotion_count = enki_pickle.get_emotion_counts()
    # enki_utterances = enki_pickle.get_utterances()
    # print(len(enki_utterances))

    # combined_ses = visualize_emotion_counts(enki_emotion_count, image_save_path)
    # write_to_csv(enki_emotion_count, csv_save_path, False)
    # write_to_csv(combined_ses, csv_save_path, True)
    # plot_roc('../Models/Enki_IEMOCAP/Enki_IEMOCAP_Various_Sentiments.csv','../')

if __name__ == "__main__":
    main()

