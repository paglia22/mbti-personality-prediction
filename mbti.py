import pandas as pd
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import random
from sklearn import svm
from keras.optimizers import Adam
from keras.layers import LeakyReLU
from nltk.stem import WordNetLemmatizer
import operator
from textblob import TextBlob
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
import re
from wordcloud import WordCloud
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

class MBTI():
    def __init__(self):
        self.csv_path = "mbti_1.csv"
        self.df = pd.read_csv(self.csv_path)
        self.original_df = self.df.copy()
        self.porter = PorterStemmer()
        self.lancaster = LancasterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.all_words = {}

    def store_clean_df(self):
        self.df.to_csv('clean.csv')

    def load_clean_df(self):
        self.df = pd.read_csv('clean.csv')

    def transform_df(self):
        # Transform the df into four different df - one for each subproblem (IE,JP,NS,TF)
        transformed_df = self.df.copy()
        transformed_df['posts'] = transformed_df['posts'].apply(lambda x: x.replace('|||', ''))
        transformed_df['posts'] = transformed_df['posts'].apply(lambda x: ''.join([i for i in x if not i.isdigit()]))
        counter = 0
        print(transformed_df.size)
        transformed_df['posts'] = transformed_df.apply(lambda row: nltk.word_tokenize(row['posts']), axis=1)
        for row_posts in transformed_df['posts'].tolist():
            print(counter)
            print(row_posts)
            counter+=1
            for feature in row_posts:
                try:
                    self.all_words[feature] += 1
                except:
                    self.all_words[feature] = 0
        print('Features found')
        self.all_words = dict(sorted(self.all_words.items(), key=operator.itemgetter(1), reverse=True))
        keys = list(self.all_words.keys())[:5000]
        exists = {}
        counter = 0
        for word in keys:
            counter +=1
            print(counter)
            exists[word] = []
            for row_posts in transformed_df['posts'].tolist():
                features = row_posts
                exists[word].append(features.count(word))
        for word in exists:
            transformed_df[word]= exists[word]
        del transformed_df['type']
        del transformed_df['posts']
        IE_df = transformed_df.copy()
        del IE_df['JP']
        del IE_df['TF']
        del IE_df['NS']
        del IE_df['Unnamed: 0']
        JP_df = transformed_df.copy()
        del JP_df['IE']
        del JP_df['TF']
        del JP_df['NS']
        del JP_df['Unnamed: 0']
        TF_df = transformed_df.copy()
        del TF_df['JP']
        del TF_df['IE']
        del TF_df['NS']
        del TF_df['Unnamed: 0']
        NS_df = transformed_df.copy()
        del NS_df['JP']
        del NS_df['IE']
        del NS_df['TF']
        del NS_df['Unnamed: 0']
        print('Finished')
        return IE_df, JP_df, TF_df, NS_df

    def post_cleaner(self, post):
        post = post.lower()
        post = re.sub(
            r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''',
            '', post, flags=re.MULTILINE)
        puncs1 = ['@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '_', '+', '=', '{', '}', '[', ']', '\\', '"',
                  "'", ';', ':', '<', '>', '/']

        for punc in puncs1:
            post = post.replace(punc, '')

        puncs2 = [',', '.', '?', '!', '\n']
        for punc in puncs2:
            post = post.replace(punc, ' ')

        post = re.sub('\s+', ' ', post).strip()

        return post

    def perform_eda(self):
        # ++++++ Print information and description of the data
        #print("+++++++++++ self.df.info:")
        print(self.df.info())

        types = self.df.type.tolist()
        pd.Series(types).value_counts().plot(kind="bar")
        plt.savefig("plot1.png")

    def stemSentence(self, sentence):
        token_words = word_tokenize(sentence)
        stem_sentence = []
        for word in token_words:
            stem_sentence.append(self.lemmatizer.lemmatize(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)

    def prepare_df(self):
        posts = self.df.posts.tolist()
        #clean
        posts = [self.post_cleaner(post) for post in posts]
        #lemmatize
        posts = [self.stemSentence(post) for post in posts]
        self.df['posts'] = posts

        #print(self.df.head(1))
        # Create 4 more columns for binary classification - LABEL ENCODING, ONE-HOT ENCODING
        map1 = {"I": 0, "E": 1}
        map2 = {"N": 0, "S": 1}
        map3 = {"T": 0, "F": 1}
        map4 = {"J": 0, "P": 1}
        self.df['IE'] = self.df['type'].astype(str).str[0]
        self.df['IE'] = self.df['IE'].map(map1)
        self.df['NS'] = self.df['type'].astype(str).str[1]
        self.df['NS'] = self.df['NS'].map(map2)
        self.df['TF'] = self.df['type'].astype(str).str[2]
        self.df['TF'] = self.df['TF'].map(map3)
        self.df['JP'] = self.df['type'].astype(str).str[3]
        self.df['JP'] = self.df['JP'].map(map4)

    def add_features(self):
        # Add new features, such as words per comment, links per comment, images per comment...
        self.df['ellipsis_per_comment'] = self.df['posts'].apply(lambda x: x.count('...') / (x.count("|||") + 1))
        self.df['words_per_comment'] = self.df['posts'].apply(lambda x: x.count(' ') / (x.count("|||") + 1))
        self.df['words'] = self.df['posts'].apply(lambda x: x.count(' '))
        self.df['link_per_comment'] = self.df['posts'].apply(lambda x: x.count('http') / (x.count("|||") + 1))
        self.df['smiles_per_comment'] = self.df['posts'].apply(lambda x: (x.count(':-)') + x.count(':)') + x.count(':-D') + x.count(':D')) / (x.count("|||") + 1))
        self.df['sad'] = self.df['posts'].apply(lambda x: (x.count(':(') + x.count('):') ) / (x.count("|||") + 1))
        self.df['heart'] = self.df['posts'].apply(lambda x: x.count('<3') / (x.count("|||") + 1))
        self.df['smiling'] = self.df['posts'].apply(lambda x: x.count(';)') / (x.count("|||") + 1))
        self.df['exclamation_mark_per_comment'] = self.df['posts'].apply(lambda x: x.count("!") / (x.count("|||") + 1))
        self.df['question_mark_per_comment'] = self.df['posts'].apply(lambda x: x.count("?") / (x.count("|||") + 1))
        self.df['polarity'] = self.df['posts'].apply(lambda x: TextBlob(x).sentiment.polarity)

    def plot(self):
        # Plot each category to see if it is balanced - We observe that IE and NS are fairly imbalanced.
        binary1 = self.df.IE.tolist()
        pd.Series(binary1).value_counts().plot(kind="bar", title="0=I, 1=E")
        # plt.show()
        plt.savefig("IE.png")

        binary1 = self.df.NS.tolist()
        pd.Series(binary1).value_counts().plot(kind="bar", title="0=N, 1=S")
        # plt.show()
        plt.savefig("NS.png")

        binary1 = self.df.TF.tolist()
        pd.Series(binary1).value_counts().plot(kind="bar", title="0=T, 1=F")
        # plt.show()
        plt.savefig("TF.png")

        binary1 = self.df.JP.tolist()
        pd.Series(binary1).value_counts().plot(kind="bar", title="0=J, 1=P")
        # plt.show()
        plt.savefig("JP.png")

        # PLOT 2
        plt.figure(figsize=(15, 10))
        sns.swarmplot("type", "words_per_comment", data=self.df)
        plt.savefig("plot2.png")

        # PLOT 3
        plt.figure(figsize=(15, 10))
        sns.jointplot("variance_of_word_counts", "words_per_comment", data=self.df, kind="hex")
        # plt.show()
        plt.savefig("plot3.png")

    def wordcloud(self):
        fig, ax = plt.subplots(len(self.df['type'].unique()), sharex=True, figsize=(15,10*len(self.df['type'].unique())))
        k = 0
        for i in self.df['type'].unique():
            df_4 = self.df[self.df['type'] == i]
            wordcloud = WordCloud().generate(df_4['posts'].to_string())
            ax[k].imshow(wordcloud)
            ax[k].set_title(i)
            ax[k].axis("off")
            k+=1
        wordcloud.to_file('N.png')

    def create_clean_df(self):
        self.perform_eda()
        self.add_features()
        self.prepare_df()
        self.store_clean_df()

    def create_transformed_df(self):
        self.load_clean_df()
        IE_df, JP_df, TF_df, NS_df = self.transform_df()
        IE_df.to_csv('IE_df.csv')
        JP_df.to_csv('JP_df.csv')
        TF_df.to_csv('TF_df.csv')
        NS_df.to_csv('NS_df.csv')

    def remove_bars(self):
        self.df['posts'] = self.df['posts'].apply(lambda x: x.replace('|||', ''))

    def svm(self):
        IE_df = pd.read_csv('IE_df.csv')
        y = IE_df['IE']
        del IE_df['IE']
        x_train, x_test, y_train, y_test = train_test_split(IE_df, y, test_size=0.20, random_state=1, stratify=y)
        IE_accuracy = self.perform_svm(x_train, x_test, y_train, y_test)
        print('IE')
        print(IE_accuracy)

        JP_df = pd.read_csv('JP_df.csv')
        y = JP_df['JP']
        del JP_df['JP']
        x_train, x_test, y_train, y_test = train_test_split(JP_df, y, test_size=0.20, random_state=1, stratify=y)
        JP_accuracy = self.perform_svm(x_train, x_test, y_train, y_test)
        print('JP')
        print(JP_accuracy)

        TF_df = pd.read_csv('TF_df.csv')
        y = TF_df['TF']
        del TF_df['TF']
        x_train, x_test, y_train, y_test = train_test_split(TF_df, y, test_size=0.20, random_state=1, stratify=y)
        TF_accuracy = self.perform_svm(x_train, x_test, y_train, y_test)
        print('TF')
        print(TF_accuracy)

        NS_df = pd.read_csv('NS_df.csv')
        y = NS_df['NS']
        del NS_df['NS']
        x_train, x_test, y_train, y_test = train_test_split(NS_df, y, test_size=0.20, random_state=1, stratify=y)
        NS_accuracy = self.perform_svm(x_train, x_test, y_train, y_test)
        print('NS')
        print(NS_accuracy)

    def lsmt(self, type, dropout, max_words, max_len, neurons):
        self.load_clean_df()
        X = self.df['posts']
        Y = self.df[type]
        le = LabelEncoder()
        Y = le.fit_transform(Y)
        Y = Y.reshape(-1, 1)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=1, stratify=Y)
        tok = Tokenizer(num_words=max_words)
        tok.fit_on_texts(X_train)
        sequences = tok.texts_to_sequences(X_train)
        sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
        inputs = Input(name='inputs', shape=[max_len])
        layer = Embedding(max_words, 50, input_length=max_len)(inputs)
        layer = LSTM(64)(layer)
        layer = Dense(neurons, name='FC1')(layer)
        layer = Dense(neurons, name='FC2')(layer)
        layer = Dense(neurons, name='FC3')(layer)
        layer = LeakyReLU(alpha=0.1)(layer)
        layer = Dropout(dropout)(layer)
        layer = Dense(1, name='out_layer')(layer)
        layer = Activation('sigmoid')(layer)
        model = Model(inputs=inputs, outputs=layer)
        model.summary()
        model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
        model.fit(sequences_matrix, Y_train, batch_size=128, epochs=10,
                  validation_split=0.2, callbacks=[EarlyStopping(monitor='val_loss', min_delta=0.0001)])
        test_sequences = tok.texts_to_sequences(X_test)
        test_sequences_matrix = sequence.pad_sequences(test_sequences, maxlen=max_len)
        accr = model.evaluate(test_sequences_matrix, Y_test)
        with open('report.txt', 'a') as file:
            file.write(f'Type: {type}\nDropout: {dropout} \nMax words: {max_words} \nMax len: {max_len} \nNeurons: {neurons}  Loss: {accr[0]}\n  Accuracy: {accr[1]}\n\n\n\n')

    def try_random_combinations_lstm(self, type):
        dropouts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        max_words_p = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
        max_lens = [50, 100, 150, 200, 250, 500, 750, 1000]
        neurons_p = [256, 300, 350, 400]
        tried = []
        while (True):
            dropout = random.choice(dropouts)
            max_words = random.choice(max_words_p)
            max_len = random.choice(max_lens)
            neurons = random.choice(neurons_p)
            identifier = f"{dropout}{max_words}{max_lens}{neurons}"
            if identifier not in tried:
                self.lsmt(type=type, dropout=dropout, neurons=neurons, max_words=max_words, max_len=max_len)
                tried.append(identifier)

    def perform_svm(x_train, x_test, y_train, y_test):
        scaling = MinMaxScaler(feature_range=(-1, 1)).fit(x_train)
        x_train = scaling.transform(x_train)
        x_test = scaling.transform(x_test)
        clf = svm.SVC(C=10, kernel='linear', degree=1, gamma='auto')
        clf.fit(x_train, y_train)
        #scores = cross_val_score(clf, x_train, x_test, cv=10)
        score = clf.score(x_test, y_test)
        return score

def main():
    mbti = MBTI()

    #If you do not have a clean csv
    #mbti.create_clean_df()

    # SVM
    #If you do not have transformed datasets
    #mbti.create_transformed_df()
    #mbti.svm()

    #LSTM
    mbti.lsmt('IE', dropout=0.5, max_words=1600, max_len=50, neurons=350)

main()

