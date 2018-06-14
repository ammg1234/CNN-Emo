# coding=utf8

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import string
import re
from collections import Counter
from nltk.corpus import stopwords
# from sklearn.model_selection import train_test_split
import spacy
import os
import time
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding

spacy.load('en_core_web_sm')
from spacy.lang.en import English
#vocabulary_size = 20000

import  argparse

parser = argparse.ArgumentParser()
parser.add_argument("-voc", "--voc_size", help="Entrada de dimensión para vocabulario de datos [defecto 20000]", default=20000)
parser.add_argument("-vocout", "--vocout_size", help="Salida de dimensión para vocabulario de datos [defecto 280]", default=280)
parser.add_argument("-ep", "--epoca", help="Número de epocas para entrenar la red [defecto 3]", default=3)
parser.add_argument("-cw", "--comwords", help="Mostrar gráfica de palabras más comunes [n = no, y = si]", default="y")
parser.add_argument("-cl", "--classes", help="Mostrar gráfica de total clases [n = no, y = si]", default="y")
parser.add_argument("-fl", "--filters", help="Tamaño de filtro para la red [defecto 128]", default=128)
parser.add_argument("-krn", "--kernel", help="Longitud de ventana para la red [defecto 4]", default=4)
parser.add_argument("-hl", "--hidlay", help="Número de neuronals ocultas para la red [defecto 1]", default=1)

args = parser.parse_args()

def tokenizeText(sample):
    tokens = parser(sample)
    lemmas = []
    for tok in tokens:
        lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)
    tokens = lemmas
    tokens = [tok for tok in tokens if tok not in STOPLIST]
    tokens = [tok for tok in tokens if tok not in SYMBOLS]
    return tokens

def create_conv_model():
    model_conv = Sequential()
    model_conv.add(Embedding(vocabulary_size, args.vocout_size, input_length=max(length)))
    #model_conv.add(Embedding(vocabulary_size, 100, input_length=max(length)+1))
    # model_conv.add(Dropout(0.1))
    model_conv.add(Conv1D(args.filters, args.kernel, activation='relu'))
    model_conv.add(MaxPooling1D())
    model_conv.add(LSTM(100))
    model_conv.add(Dense(args.hidlay, activation='sigmoid'))
    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv

def rem_user(tuit):
    tuit = tuit.replace('@', '').replace("#", "").strip()
    return tuit


def cleanup_text(docs, emoc):
    texts = []
    counter = 1
    for doc in docs:
        print(str(counter) + " de " + str(len(docs)) + " oraciones procesadas [" + emoc + "].")
        counter += 1
        doc = nlp(doc, disable=['parser', 'ner'])
        tokens = [tok.lemma_.lower().strip() for tok in doc if tok.lemma_ != '-PRON-']
        tokens = [tok for tok in tokens if tok not in set(
            stopwords.words('english') + list(ENGLISH_STOP_WORDS) + list(punctuations) + ['...', 'amp'])]
        tokens = ' '.join(tokens)
        texts.append(tokens)
    return pd.Series(texts)


def common_words(emoc_text, emoc):
    text_clean = cleanup_text(emoc_text, emoc)
    text_clean = ' '.join(text_clean).split()
    emoc_counts = Counter(text_clean)
    emoc_common_words = [word[0] for word in emoc_counts.most_common(5)]
    emoc_common_counts = [word[1] for word in emoc_counts.most_common(5)]
    fig = plt.figure()
    sns.barplot(x=emoc_common_words, y=emoc_common_counts)
    # plt.Text('Holi')
    fig.canvas.set_window_title("Palabras más comunes: " + emoc)
    plt.show()


if __name__ == "__main__":
    # stopwords = stopwords.words('english')
    lista_ora_train = list()
    lista_cla_train = list()
    lista_ora_test = list()
    lista_cla_test = list()
    # datos_anger = datos_joy = datos_fear = datos_sadness = []
    dir_train = '//home//angelomarlon//Documentos//Training, Develop And Test//02 EI-oc//Train//txt-En'
    dir_test = '//home//angelomarlon//Documentos//Training, Develop And Test//02 EI-oc//Test//txt-En'
    clase_num = -1
    for filename in os.listdir(dir_train):
        with open(dir_train + "//" + filename, 'r') as f:
            emoc = filename.split('-')[3]
            lista = list()
            for linea in f:
                linea = linea.split('\t')
                if linea[0] != "ID":
                    if emoc == "sadness":
                        clase_num = 0
                    elif emoc == "joy":
                        clase_num = 1
                    elif emoc == "anger":
                        clase_num = 2
                    elif emoc == "fear":
                        clase_num = 3
                    lista_cla_train.append(clase_num)
                    lista_ora_train.append(rem_user(linea[1]))

    for filename in os.listdir(dir_test):
        with open(dir_test + "//" + filename, 'r') as f:
            emoc = filename.split('-')[4]
            lista = list()
            for linea in f:
                linea = linea.split('\t')
                if linea[0] != "ID":
                    if emoc == "sadness":
                        clase_num = 0
                    elif emoc == "joy":
                        clase_num = 1
                    elif emoc == "anger":
                        clase_num = 2
                    elif emoc == "fear":
                        clase_num = 3
                    lista_cla_test.append(clase_num)
                    lista_ora_test.append(rem_user(linea[1]))

    print("Número de oraciones en datos de entrenamiento:\t", len(lista_ora_train))
    print("Número de oraciones en datos de prueba:\t", len(lista_ora_test))

    df_train = pd.DataFrame(columns=['oracion', 'clase'])
    df_test = pd.DataFrame(columns=['oracion', 'clase'])

    lc_train = pd.Series(lista_cla_train)
    lo_train = pd.Series(lista_ora_train)

    lc_test = pd.Series(lista_cla_test)
    lo_test = pd.Series(lista_ora_test)

    df_train['oracion'] = lo_train.values
    df_train['clase'] = lc_train.values

    df_test['oracion'] = lo_test.values
    df_test['clase'] = lc_test.values

    # df = pd.DataFrame.from_dict()
    # df = pd.read_csv('//home//angelomarlon//Descargas//research_paper.csv')

    # print('Título de clase de emoción:', train['clase'].iloc[0])
    # print('Oración de clase:', train['oracion'].iloc[0])
    print('Forma de los datos de entrenamiento:', df_train.shape)
    print('Forma de los datos de prueba:', df_test.shape)

    if args.classes.lower() == "y":
        fig = plt.figure()
        sns.barplot(x=df_train['clase'].unique(), y=df_train['clase'].value_counts())
        fig.canvas.set_window_title('Emociones - Train')
        plt.show()

        fig = plt.figure()
        sns.barplot(x=df_test['clase'].unique(), y=df_test['clase'].value_counts())
        fig.canvas.set_window_title('Emociones - Test')
        plt.show()

    nlp = spacy.load('en_core_web_sm')
    punctuations = string.punctuation

    if args.comwords.lower() == "y":
        Sadness_text = [text for text in df_train[df_train['clase'] == 0]['oracion']]
        common_words(Sadness_text, 'sadness')
        Joy_text = [text for text in df_train[df_train['clase'] == 1]['oracion']]
        common_words(Joy_text, 'joy')
        Anger_text = [text for text in df_train[df_train['clase'] == 2]['oracion']]
        common_words(Anger_text, 'anger')
        Fear_text = [text for text in df_train[df_train['clase'] == 3]['oracion']]
        common_words(Fear_text, 'fear')

    input('Enter...')

    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score
    from nltk.corpus import stopwords
    import string
    import re
    import spacy

    spacy.load('en_core_web_sm')
    from spacy.lang.en import English

    parser = English()

    STOPLIST = set(stopwords.words('english') + list(ENGLISH_STOP_WORDS))
    SYMBOLS = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”"]


    # data
    train1 = df_train['oracion'].tolist()
    labelsTrain1 = df_train['clase'].tolist()

    test1 = df_test['oracion'].tolist()
    labelsTest1 = df_test['clase'].tolist()


    # df_train['oracion'] = df_train['oracion'].map(lambda x: tokenizeText(x))

    vocabulary_size = int(args.voc_size)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(df_train['oracion'])
    sequences = tokenizer.texts_to_sequences(df_train['oracion'])

    length = []
    for x in df_train['oracion']:
        length.append(len(x.split()))

    data = pad_sequences(sequences, maxlen=max(length))

    model_conv = create_conv_model()

    #model_ptw2v.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=5, batch_size=32, verbose=2)

    model_conv.fit(data, labelsTrain1, epochs=args.epoca)

    tokenizer_test = Tokenizer()
    tokenizer_test.fit_on_texts(df_test['oracion'])
    sequences_test = tokenizer_test.texts_to_sequences(df_test['oracion'])
    data_test = pad_sequences(sequences_test, maxlen=max(length))

    preds = model_conv.predict(data_test)

    # train
    # pipe.fit(train1, labelsTrain1)

    # test
    #pipe.predict(test1)
    print("accuracy:", accuracy_score(labelsTest1, preds))

    df_save = pd.DataFrame(data)
    df_label = pd.DataFrame(np.array(labelsTrain1))

    result = pd.concat([df_save, df_label])

    filename_csv = 'dense_word_vectors '+ time.strftime("%d_%m_%y_%H_%M_%S")  +'.csv'

    result.to_csv(filename_csv, index=False)

    print("Resultados de vectores arrojados por la red encontrados en: " + filename_csv)

    #print("Top 10 features used to predict: ")

    #printNMostInformative(vectorizer, clf, 10)

    #pipe = Pipeline([('cleanText', CleanTextTransformer()), ('vectorizer', vectorizer)])
    #transform = pipe.fit_transform(train1, labelsTrain1)
    #vocab = vectorizer.get_feature_names()

    #for i in range(len(train1)):
    #    s = ""
    #    indexIntoVocab = transform.indices[transform.indptr[i]:transform.indptr[i + 1]]
    #    numOccurences = transform.data[transform.indptr[i]:transform.indptr[i + 1]]
    #    for idx, num in zip(indexIntoVocab, numOccurences):
    #        s += str((vocab[idx], num))

    #from sklearn import metrics

    #print(metrics.classification_report(np.array(labelsTest1), preds))
