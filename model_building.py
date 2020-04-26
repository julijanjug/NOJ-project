import pandas as pd
import numpy as np
import spacy
from tqdm import tqdm
import re
import time
import pickle
import tensorflow_hub as hub
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

#naredi elmo vector za words_before združene z presledki kr za vse stavke ki so povezani s to entitetio en vector

print("importing elmo")
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
print("done")

def elmo_vectors(x):
    x = x.values
    x = [' '.join(i) for i in x]
    print(x)
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))

def load_preprocess_data(limit=None):
    #load data
    data = np.load("data/data.npy", allow_pickle=True)
    data = np.delete(data, (0), axis=0)
    data = data[data[:,4] != None]  #removing None sentiments
    data[:,4] = np.where(data[:,4] == '5', '4', data[:,4]) #change class 5->4
    data[:,4] = np.where(data[:,4] == '1', '2', data[:,4]) #change class 1->2
    data = pd.DataFrame(data)

    if limit != None:
        data = data[0:limit]

    #sentiment distribution
    print(data[4].value_counts(normalize = True))

    #text preprocesing
    # convert text to lowercase
    data[5] = data[5].apply(lambda x: [s.lower() for s in x])
    # remove numbers
    data[5] = data[5].apply(lambda x: [s.replace("[0-9]", " ") for s in x])
    # remove whitespaces
    data[5] = data[5].apply(lambda x: [' '.join(s.split()) for s in x])

    #train test split
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    return train, test

def make_elmo_embeddings(train, test):
    print("elmo is eating his cookie...")
    # split vector into embedings for speed
    list_train = [train[i:i+100] for i in range(0,train.shape[0],100)]
    list_test = [test[i:i+100] for i in range(0,test.shape[0],100)]
    # Extract ELMo embeddings
    elmo_train = [elmo_vectors(x[5]) for x in list_train]
    elmo_test = [elmo_vectors(x[5]) for x in list_test]
    #join them back together
    elmo_train_new = np.concatenate(elmo_train, axis = 0)
    elmo_test_new = np.concatenate(elmo_test, axis = 0)
    print("done")

    # save elmo_train_new
    pickle_out = open("elmo_embeddings/elmo_train_upsampled.pickle", "wb")
    pickle.dump(elmo_train_new, pickle_out)
    pickle_out.close()
    # save elmo_test_new
    pickle_out = open("elmo_embeddings/elmo_test_upsampled.pickle", "wb")
    pickle.dump(elmo_test_new, pickle_out)
    pickle_out.close()

def load_elmo_embeddings():
    # load elmo_train_new
    pickle_in = open("elmo_embeddings/elmo_train_upsampled.pickle", "rb")
    elmo_train_new = pickle.load(pickle_in)
    # load elmo_train_new
    pickle_in = open("elmo_embeddings/elmo_test_upsampled.pickle", "rb")
    elmo_test_new = pickle.load(pickle_in)
    return elmo_train_new, elmo_test_new

def upsample_minority(df):
    df_majority = df.loc[df[4] == '3']
    df_minority_2 = df.loc[df[4] == '2']
    df_minority_4 = df.loc[df[4] == '4']
    df_minority = pd.concat([df_minority_2, df_minority_4])

    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=42)  # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled

def fit_log_reg(elmo_train_new, train, elmo_test_new, test):
    #training logistinc regresion
    print("training log reg...")
    lreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', dual=False, max_iter=1000, random_state=23)
    lreg.fit(elmo_train_new, train[4])
    print("done")

    # make predictions on test set
    preds_test = lreg.predict(elmo_test_new)
    print("F1 test: ", f1_score(test[4], preds_test, average='micro'))

    #test accuracy
    print("CA: ", accuracy_score(test[4], preds_test))

    #recall and ROC area
    print('Recall score: {}'.format(recall_score(test[4], preds_test, average='micro')))

    #confusion matrix
    labels = ['2', '3', '4']
    print(confusion_matrix(test[4], preds_test, labels))

    # training random forest
    print("training random forest...")
    rand_forest = RandomForestClassifier(n_estimators=50, random_state=69)
    rand_forest.fit(elmo_train_new, train[4])
    preds = rand_forest.predict(elmo_test_new)
    print("Random Forest Classification Score: ", rand_forest.score(elmo_test_new, test[4]))
    # confusion matrix
    labels = ['2', '3', '4']
    print(confusion_matrix(test[4], preds, labels))

#-------MAIN------
train, test = load_preprocess_data(800)

# upsample minority class in train dataset
train = upsample_minority(train)
print(train[4].value_counts(normalize=True))

# make_elmo_embeddings(train, test)
elmo_train_new, elmo_test_new = load_elmo_embeddings()
fit_log_reg(elmo_train_new, train, elmo_test_new, test)


