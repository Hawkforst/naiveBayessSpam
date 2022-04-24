import sklearn.metrics as sklearn_metrics
import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation

import numpy as np
from collections import Counter
import warnings

en_sm_model = spacy.load("en_core_web_sm")
warnings.filterwarnings("ignore")


class ConfusionMatrix:
    def __init__(self):
        self.TN = 0
        self.FP = 0
        self.FN = 0
        self.TP = 0

    def get_total(self):
        return self.TN + self.TP + self.FN + self.FP


class MetricStorage:
    def __init__(self):
        self.accuracy = 0.0
        self.recall = 0.0
        self.precision = 0.0
        self.F1 = 0.0


class DataProcessor:

    def __init__(self):
        self.accuracy = 0.0
        self.train_set = pd.DataFrame
        self.test_set = pd.DataFrame
        self.vocabulary = list()
        self.test_vocabulary = list()
        self.data = pd.read_csv('spam.csv', encoding='iso-8859-1')
        self.__cleanup_data()
        self.word_probabilities = pd.DataFrame(columns=['Spam Probability', 'Ham Probability'])
        self.lapl_smooth = 1
        self.prediction = pd.DataFrame
        self.metrics = MetricStorage()
        self.confusion_matrix = ConfusionMatrix()
        self.bag_of_words_data = None
        self.bag_of_words_data_target = None
        self.bag_of_words_test = None
        self.bag_of_words_test_target = None

    def get_data_rows(self):
        return self.train_set.shape[0]

    def get_data_cols(self):
        return self.train_set.shape[1]

    def __cleanup_data(self):
        self.data.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace=True)
        self.data.columns = ['Target', 'SMS']
    
    def process_data(self):
        for index, row in self.data.iterrows():
            self.data.at[index, 'SMS'] = self.process_text(row['SMS'])

    def process_text(self, text):
        text = text.lower()
        doc = en_sm_model(text)
        new_text = []
        for token in doc:
            if any(ch.isdigit() for ch in token.text):
                new_text.append("aanumbers")
            elif len(token.lemma_) > 1 and token.lemma_ not in STOP_WORDS and not any(
                    ch in punctuation for ch in token.text):
                new_text.append(token.lemma_)
        return ' '.join(new_text)

    def display_data(self, max_rows=200, max_cols=2, _data='train'):
        pd.options.display.max_columns = max_cols
        pd.options.display.max_rows = max_rows
        show_all = True if _data == 'all' else False
        if _data == 'train' or show_all:
            print(self.train_set.iloc[:max_rows, :max_cols])
        if _data == 'prob' or show_all:
            print(self.word_probabilities.iloc[:max_rows, :2])
        if _data == 'pred' or show_all:
            print(self.prediction.iloc[:50, :2])
        if _data == 'metrics' or show_all:
            print({'Accuracy': self.metrics.accuracy,
                    'Recall': self.metrics.recall,
                    'Precision': self.metrics.precision,
                    'F1': self.metrics.F1})

    def split_data(self):
        train_last_index = int(self.data.shape[0] * 0.8)
        self.data = self.data.sample(frac=1, random_state=43)
        self.train_set = self.data[0:train_last_index]
        self.test_set = self.data[train_last_index:]

    def build_vocabulary(self,test=False):
        if not test:
            for index, row in self.train_set.iterrows():
                self.vocabulary.extend(list(set(row['SMS'].split())))
            self.vocabulary = sorted(list(set(self.vocabulary)))
        else:
            for index, row in self.test_set.iterrows():
                self.test_vocabulary.extend(list(set(row['SMS'].split())))
            self.test_vocabulary = sorted(list(set(self.test_vocabulary)))

    def bag_of_words(self, test=False):
        self.split_data()
        if not test:
            self.train_set = self.train_set.reset_index(drop=True)
            self.build_vocabulary()
            self.bag_of_words_data = np.zeros(shape=(self.get_data_rows(), len(self.vocabulary)), dtype=int)
            self.bag_of_words_data_target = np.zeros(len(self.vocabulary))
            for index, row in self.train_set.iterrows():
                word_freq = Counter(row['SMS'].split())
                self.bag_of_words_data_target[index] = 1 if row['Target'] == 'spam' else 0
                for word, freq in word_freq.items():
                    self.bag_of_words_data[index, self.vocabulary.index(word)] = freq
            for idx, term in enumerate(self.vocabulary):
                self.train_set[term] = self.bag_of_words_data[:, idx]
        else:
            self.test_set = self.test_set.reset_index(drop=True)
            self.build_vocabulary(test=True)
            self.bag_of_words_test = np.zeros(shape=(self.get_data_rows(), len(self.test_vocabulary)), dtype=int)
            self.bag_of_words_test_target = np.zeros(len(self.test_vocabulary))
            for index, row in self.test_set.iterrows():
                word_freq = Counter(row['SMS'].split())
                self.bag_of_words_test_target[index] = 1 if row['Target'] == 'spam' else 0
                for word, freq in word_freq.items():
                    self.bag_of_words_test[index, self.test_vocabulary.index(word)] = freq
            #for idx, term in enumerate(self.test_vocabulary):
            #    self.test_set[term] = self.bag_of_words_test[:, idx]

    def calc_word_probabilities(self):
        n_spam = {}; n_ham = {}; n_tot = {}
        total_ham = 0
        total_spam = 0
        for item in self.train_set.iloc[0].keys():
            if not (item == 'Target' or item == 'SMS'):
                n_spam[item], n_ham[item], n_tot[item] = 0, 0, 0
        for index, row in self.train_set.iterrows():
            is_spam = True if row['Target'] == 'spam' else False
            for item, freq in zip(row.keys(), row):
                if item == 'Target' or item == 'SMS':
                    pass
                else:
                    if is_spam:
                        n_spam[item] += freq
                        total_spam += freq
                    else:
                        n_ham[item] += freq
                        total_ham += freq
                    n_tot[item] += freq

        totalVocab = len(n_tot)
        for item in self.train_set.iloc[0].keys():
            if not (item == 'Target' or item == 'SMS'):
                probability_spam = self.contidional_probability(total_spam, n_spam[item], totalVocab)
                probability_ham = self.contidional_probability(total_ham, n_ham[item], totalVocab)
                self.word_probabilities.loc[item] = [probability_spam, probability_ham]

    def contidional_probability(self, n_word, n_event, n_vocab):
        return (n_event + self.lapl_smooth) / (n_word + self.lapl_smooth * n_vocab)

    def classify_sentence(self, dataframe):
        sms = dataframe['SMS']
        prob_spam = 1.0
        prob_ham = 1.0
        sentence = self.process_text(sms)
        for word in sentence.split(' '):
            if word in self.word_probabilities.index:
                prob_ham *= self.word_probabilities.loc[word]['Ham Probability']
                prob_spam *= self.word_probabilities.loc[word]['Spam Probability']
        prediction = 'spam' if prob_spam > prob_ham else 'ham' if prob_spam < prob_ham else 'unknown'
        return prediction, dataframe['Target']

    def classifier_function(self):
        predicted_data = self.test_set.apply(self.classify_sentence, axis=1)
        self.prediction = pd.DataFrame(((i[0], i[1]) for i in predicted_data), index=self.test_set.index,
                                       columns=['Predicted', 'Actual'])

    def calculate_confusion_matrix(self):
        for row in self.prediction.iterrows():
            if row[1]['Predicted'] == 'spam' and row[1]['Actual'] == 'spam':
                self.confusion_matrix.TN += 1
            if row[1]['Predicted'] == 'ham' and row[1]['Actual'] == 'spam':
                self.confusion_matrix.FP += 1
            if row[1]['Predicted'] == 'ham' and row[1]['Actual'] == 'ham':
                self.confusion_matrix.TP += 1
            if row[1]['Predicted'] == 'ham' and row[1]['Actual'] == 'spam':
                self.confusion_matrix.FP += 1

    def calculate_accuracy(self):
        self.metrics.accuracy = (self.confusion_matrix.TP + self.confusion_matrix.TN) / self.confusion_matrix.get_total()

    def calculate_recall(self):
        self.metrics.recall = self.confusion_matrix.TP / (self.confusion_matrix.TP + self.confusion_matrix.FN)

    def calculate_precision(self):
        self.metrics.precision = self.confusion_matrix.TP / (self.confusion_matrix.TP + self.confusion_matrix.FP)

    def calculate_F1(self):
        self.metrics.F1 = 2 * self.metrics.precision * self.metrics.recall / (
                    self.metrics.precision + self.metrics.recall)

    def calculate_metrics(self):
        self.calculate_confusion_matrix()
        self.calculate_accuracy()
        self.calculate_recall()
        self.calculate_precision()
        self.calculate_F1()




data = DataProcessor()
data.process_data()
data.bag_of_words()
data.calc_word_probabilities()
data.classifier_function()
data.calculate_metrics()

data.process_data()
data.bag_of_words(test=True)


#data.display_data(200, 50, _data='metrics')


clf = MultinomialNB()
clf.fit(data.bag_of_words_data.transpose(),
        np.transpose(data.bag_of_words_data_target))
sklearn_prediction = clf.predict(data.bag_of_words_test.transpose())

sk_metrics = MetricStorage()

sk_metrics.accuracy = sklearn_metrics.accuracy_score(sklearn_prediction, data.bag_of_words_test_target)
sk_metrics.recall = sklearn_metrics.recall_score(sklearn_prediction, data.bag_of_words_test_target)
sk_metrics.precision = sklearn_metrics.precision_score(sklearn_prediction, data.bag_of_words_test_target)
sk_metrics.F1 = sklearn_metrics.f1_score(sklearn_prediction, data.bag_of_words_test_target)

print({'Accuracy': sk_metrics.accuracy,
       'Recall': sk_metrics.recall,
       'Precision': sk_metrics.precision,
       'F1': sk_metrics.F1})
