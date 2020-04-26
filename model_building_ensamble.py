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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from scipy import stats as s
from sklearn.utils import resample

#This file processes data that has formated words before as list of lists and makes elmo vector for each sentence.
# Then predists every sentence and takes the most frequent prediction

print("importing elmo")
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
print("done")

def elmo_vectors(x):
    # x = x.values #list of sentences
    print(x)
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))

    return elmo_vectors

def load_preprocess_data(limit=None):
    #load data
    data = np.load("data/data_v3.npy", allow_pickle=True)
    data = np.delete(data, (0), axis=0)
    data = data[data[:,4] != None]  #removing None sentiments
    data[:,4] = np.where(data[:,4] == '5', '4', data[:,4]) #change class 5->4
    data[:,4] = np.where(data[:,4] == '1', '2', data[:,4]) #change class 1->2
    data = pd.DataFrame(data)
    if limit != None: #for limiting the size of data (test purpuses)
        data = data[0:limit]

    #sentiment distribution
    print(data[4].value_counts(normalize = True))

    #text preprocesing
    # convert text to lowercase
    data[5] = data[5].apply(lambda x: [[sp.lower() for sp in s] for s in x])
    # remove numbers
    data[5] = data[5].apply(lambda x: [[sp.replace("[0-9]", " ") for sp in s] for s in x])

    #train test split
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    #expand words-before into multiple rows (every sentence in new row)
    train_new = [] #list of rows
    for i, vrstica in train.iterrows():
        if any(isinstance(el, list) for el in vrstica[5]):
            for stavek in vrstica[5]:
                train_new.append([vrstica[0]+"-"+vrstica[1], ' '.join(stavek), vrstica[4]]) #doc-ent-id, stavek-from-before,  sentiment
        else:
            train_new.append([vrstica[0]+"-"+vrstica[1], stavek, vrstica[4]]) #doc-ent-id, stavek-from-before,  sentiment
    train_new = pd.DataFrame(train_new)

    test_new = [] #list of rows
    for i, vrstica in test.iterrows():
        if any(isinstance(el, list) for el in vrstica[5]):
            for stavek in vrstica[5]:
                test_new.append([vrstica[0] + "-" + vrstica[1], ' '.join(stavek), vrstica[4]])  # doc-ent-id, stavek-from-before,  sentiment
        else:
            test_new.append([vrstica[0]+"-"+vrstica[1], stavek, vrstica[4]]) #doc-ent-id, stavek-from-before,  sentiment
    test_new = pd.DataFrame(test_new)

    return train_new, test_new

def upsample_minority(df):
    df_majority = df.loc[df[2] == '3']
    df_minority_2 = df.loc[df[2] == '2']
    df_minority_4 = df.loc[df[2] == '4']
    df_minority = pd.concat([df_minority_2, df_minority_4])

    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=len(df_majority),  # to match majority class
                                     random_state=42)  # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled

def make_elmo_embeddings(train, test):
    print("elmo is eating his cookie...")
    # split vector into embedings for speed
    list_train = [train[i:i+100] for i in range(0,train.shape[0],100)]
    list_test = [test[i:i+100] for i in range(0,test.shape[0],100)]
    # Extract ELMo embeddings
    elmo_train = [elmo_vectors(x[1]) for x in list_train]
    elmo_test = [elmo_vectors(x[1]) for x in list_test]
    #join them back together
    elmo_train_new = np.concatenate(elmo_train, axis = 0)
    elmo_test_new = np.concatenate(elmo_test, axis = 0)
    print("done")

    # save elmo_train_new
    pickle_out = open("elmo_embeddings/elmo_train_v4_Jaka.pickle", "wb")
    pickle.dump(elmo_train_new, pickle_out)
    pickle_out.close()
    # save elmo_test_new
    pickle_out = open("elmo_embeddings/elmo_test_v4_Jaka.pickle", "wb")
    pickle.dump(elmo_test_new, pickle_out)
    pickle_out.close()

def load_elmo_embeddings():
    # load elmo_train_new
    pickle_in = open("elmo_embeddings/elmo_train_v4_Jaka.pickle", "rb")
    elmo_train_new = pickle.load(pickle_in)
    # load elmo_train_new
    pickle_in = open("elmo_embeddings/elmo_test_v4_Jaka.pickle", "rb")
    elmo_test_new = pickle.load(pickle_in)
    return elmo_train_new, elmo_test_new

def format_test_sets(test, preds):
    test_new = []
    preds_new = []
    sentiments = []
    entiti_id = test[0].iloc[0] #id from first row
    last_sentiment = test[2].iloc[0] #first sentiment
    for i, row in test.iterrows():
        if row[0] == entiti_id:
            sentiments.append(preds[i])
        else:
            #add agregated sentiment row result
            test_new.append(last_sentiment)
            preds_new.append(s.mode(sentiments)[0])

            #reset variables for new entit
            sentiments = []
            entiti_id = row[0]
            last_sentiment = row[2]
            #add prediction of new entiti
            sentiments.append(preds[i])

    return test_new, preds_new


def fit_models(elmo_train_new, train, elmo_test_new, test):
    #training logistinc regresion
    print("training log reg...")
    lreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', dual=False, max_iter=1000)
    lreg.fit(elmo_train_new, train[2])
    print("done")

    # make predictions on test set
    preds_test = lreg.predict(elmo_test_new)
    formated_test, formated_preds = format_test_sets(test, preds_test)
    print("F1 test: ", f1_score(formated_test, formated_preds, average='micro'))

    #test accuracy
    print("CA: ", accuracy_score(formated_test, formated_preds))
    #confusion matrix
    print(confusion_matrix(formated_test, formated_preds))

    # training random forest
    print("training random forest...")
    rand_forest = RandomForestClassifier(n_estimators=100, random_state=69)
    rand_forest.fit(elmo_train_new, train[2])
    preds_test = rand_forest.predict(elmo_test_new)
    formated_test, formated_preds = format_test_sets(test, preds_test)
    print("Random Forest Classification Score: ", accuracy_score(formated_test, formated_preds))
    print(confusion_matrix(formated_test, formated_preds))


#-------MAIN------
train, test = load_preprocess_data(200)

# upsample minority class in train set
train = upsample_minority(train)
print(train[2].value_counts(normalize=True))

#makeelmo embedings
make_elmo_embeddings(train, test)
elmo_train_new, elmo_test_new = load_elmo_embeddings()
fit_models(elmo_train_new, train, elmo_test_new, test)


