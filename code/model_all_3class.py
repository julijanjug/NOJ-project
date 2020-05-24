import numpy as np
import numpy.ma as ma
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from nltk.corpus import state_union, stopwords
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import seaborn as sns


#2 step model building approach using elmo and sentiment counter by first classifing the polarity and then negativ/positive

print("importing elmo")
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
print("done")
lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
stop_words = set(stopwords.words('slovene'))

def load_preprocess_data(limit=None):
    #load data
    data = np.load("../data/data_v4.npy", allow_pickle=True)
    data = np.delete(data, (0), axis=0)
    data = data[data[:,4] != None]  #removing None sentiments
    # data[:,4] = np.where(data[:,4] == '5', '4', data[:,4]) #change class 5->4
    # data[:,4] = np.where(data[:,4] == '1', '2', data[:,4]) #change class 1->2
    data[:,4] = np.where(data[:,4] == '1', '1', data[:,4]) #change class 1->1
    data[:,4] = np.where(data[:,4] == '2', '1', data[:,4]) #change class 2->1
    data[:,4] = np.where(data[:,4] == '3', '0', data[:,4]) #change class 3->0
    data[:,4] = np.where(data[:,4] == '4', '2', data[:,4]) #change class 4->1
    data[:,4] = np.where(data[:,4] == '5', '2', data[:,4]) #change class 5->1
    data = pd.DataFrame(data)

    if limit != None:
        data = data[0:limit]

    #sentiment distribution
    # print(data[4].value_counts(normalize = True))

    #train test split
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    return train, test

def upsample_minority(df):
    df_majority = df.loc[df[4] == '0']
    df_minority_2 = df.loc[df[4] == '1']
    df_minority_4 = df.loc[df[4] == '2'] #samo za to da mi ni treba kode spreminjat
    df_minority = pd.concat([df_minority_2, df_minority_4])

    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=int(len(df_majority)/1.2),  # to match majority class
                                     random_state=42)  # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled

def get_sentiment_data(file, key):
    row = file[file[:, 0] == key]
    if len(row) > 0:
        result = row[0]
        if result[3] == "neutral":
            sent = 0
        elif result[3] == "positive":
            sent = 1
        elif result[3] == "negative":
            sent = 2
        return result[1], result[2], sent
    else:
        return np.nan, np.nan, np.nan

def fit_model_for_polarity(elmo_train,  train, elmo_test, test):
    #load document, paragraph, sentence data
    docData = np.load("../SentiNews/docLevel_sentiment_v1.npy", allow_pickle=True)
    docData = np.delete(docData, (0), axis=0)
    parData = np.load("../SentiNews/parLevel_sentiment_v1.npy", allow_pickle=True)
    parData = np.delete(parData, (0), axis=0)
    senData = np.load("../SentiNews/senLevel_sentiment_v1.npy", allow_pickle=True)
    senData = np.delete(senData, (0), axis=0)

    print("loading the train, test dataframes...")
    #prepare train/test in a way that lr understands it
    train_x = np.empty([12,])
    test_x = np.empty([12,])
    for index, row in train.iterrows(): #vrstice
        zeros = sum([item.count(0) for item in row[6]])
        ones  = sum([item.count(1) for item in row[6]])
        twos  = sum([item.count(2) for item in row[6]])
        key = row[0]+"-"+row[1]
        doc_avg_sen, doc_sd_sen, doc_sen = get_sentiment_data(docData, key)
        par_avg_sen, par_sd_sen, par_sen = get_sentiment_data(parData, key)
        sen_avg_sen, sen_sd_sen, sen_sen = get_sentiment_data(senData, key)
        new_row = [zeros, ones, twos, doc_avg_sen, doc_sd_sen, doc_sen, par_avg_sen, par_sd_sen, par_sen, sen_avg_sen, sen_sd_sen, sen_sen]
        new_row = [float(x) for x in new_row]
        train_x = np.vstack([train_x, new_row])
    for index, row in test.iterrows(): #vrstice
        zeros = sum([item.count(0) for item in row[6]])
        ones = sum([item.count(1) for item in row[6]])
        twos = sum([item.count(2) for item in row[6]])
        key = row[0] + "-" + row[1]
        doc_avg_sen, doc_sd_sen, doc_sen = get_sentiment_data(docData, key)
        par_avg_sen, par_sd_sen, par_sen = get_sentiment_data(parData, key)
        sen_avg_sen, sen_sd_sen, sen_sen = get_sentiment_data(senData, key)
        new_row = [zeros, ones, twos, doc_avg_sen, doc_sd_sen, doc_sen, par_avg_sen, par_sd_sen, par_sen, sen_avg_sen, sen_sd_sen, sen_sen]
        new_row = [float(x) for x in new_row]
        test_x = np.vstack([test_x, new_row])
    train_x = np.delete(train_x, (0), axis=0)
    test_x = np.delete(test_x, (0), axis=0)

    #append the elmo vector to the train/test dataframe
    # train_x = np.concatenate((train_x, elmo_train), axis=1)
    # test_x = np.concatenate((test_x, elmo_test), axis=1)
    print("done")


    #fix the nan values in train/test
    train_x = np.where(np.isnan(train_x), ma.array(train_x, mask=np.isnan(train_x)).mean(axis=1)[:, np.newaxis], train_x)
    test_x = np.where(np.isnan(test_x), ma.array(test_x, mask=np.isnan(test_x)).mean(axis=1)[:, np.newaxis], test_x)

    #majority class classificator model
    majority = DummyClassifier(strategy="most_frequent")
    majority.fit(train_x, train[4])
    preds_test = majority.predict(test_x)
    print("F1 test (majority): ", f1_score(test[4], preds_test, average='weighted'))

    # training logistinc regresion
    print("training log reg...")
    lreg = LogisticRegression(solver='lbfgs', dual=False, max_iter=10000, random_state=23)
    lreg.fit(train_x, train[4])
    print("done")

    # log reg scores
    preds_test = lreg.predict(test_x)
    print("LR F1: ", f1_score(test[4], preds_test, average='weighted'))
    print("LR CA: ", accuracy_score(test[4], preds_test))
    labels = ['1', '0', '2']
    # print(confusion_matrix(test[4], preds_test, labels))

    # training random forest
    print("training random forest...")
    rand_forest = RandomForestClassifier(n_estimators=20, random_state=69)
    rand_forest.fit(train_x, train[4])
    preds = rand_forest.predict(test_x)
    print("done")
    print("RF CA: ", rand_forest.score(test_x, test[4]))
    print("RF F1 : ", f1_score(test[4], preds, average='weighted'))
    # print(confusion_matrix(test[4], preds, labels))

def elmo_vectors(x):
    x = x.values
    x_new = []
    for list in x: #leamtiziram in odstranim stop worde
        new_row = []
        for podlist in list:
            for beseda in podlist:
                if beseda not in stop_words:
                    w = lemmatizer.lemmatize(beseda)
                    new_row.append(w)
        x_new.append(" ".join(new_row))
    # x = [' '.join(i) for i in x]
    embeddings = elmo(x_new, signature="default", as_dict=True)["elmo"]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))

def make_elmo_embeddings(train, test):
    print("elmo is eating his cookie...")
    # split vector into embedings for speed
    list_train = [train[i:i+100] for i in range(0,train.shape[0],100)]
    list_test = [test[i:i+100] for i in range(0,test.shape[0],100)]
    # Extract ELMo embeddings
    elmo_train = []
    elmo_test = []
    for i in range(len(list_train)):
        print("{}/{}".format(i, len(list_train)))
        elmo_train.append(elmo_vectors(list_train[i][5]))
    for i in range(len(list_test)):
        print("{}/{}".format(i, len(list_test)))
        elmo_test.append(elmo_vectors(list_test[i][5]))
    # elmo_train = [elmo_vectors(x[5]) for x in list_train]
    # elmo_test = [elmo_vectors(x[5]) for x in list_test]
    #join them back together
    elmo_train_new = np.concatenate(elmo_train, axis = 0)
    elmo_test_new = np.concatenate(elmo_test, axis = 0)
    print("done")

    # save elmo_train_new
    pickle_out = open("../elmo_embeddings/elmo_train_2step_v1.pickle", "wb")
    pickle.dump(elmo_train_new, pickle_out)
    pickle_out.close()
    # save elmo_test_new
    pickle_out = open("../elmo_embeddings/elmo_test_2step_v1.pickle", "wb")
    pickle.dump(elmo_test_new, pickle_out)
    pickle_out.close()

def load_elmo_embeddings():
    # load elmo_train_new
    pickle_in = open("../elmo_embeddings/elmo_train_2step_v1.pickle", "rb")
    elmo_train_new = pickle.load(pickle_in)
    # load elmo_train_new
    pickle_in = open("../elmo_embeddings/elmo_test_2step_v1.pickle", "rb")
    elmo_test_new = pickle.load(pickle_in)
    return elmo_train_new, elmo_test_new

#-------MAIN------
train, test = load_preprocess_data()

# upsample minority class in train dataset
# train = upsample_minority(train)

#elmo embeddings
# make_elmo_embeddings(train, test)
elmo_train_new, elmo_test_new = load_elmo_embeddings()

#train and output results
fit_model_for_polarity(elmo_train_new, train,  elmo_test_new, test)



