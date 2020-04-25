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
from sklearn.metrics import f1_score

print("importing elmo")
elmo = hub.Module("https://tfhub.dev/google/elmo/3", trainable=True)
print("done")

def elmo_vectors(x):
    x = x.values
    x = [' '.join(i) for i in x]
    embeddings = elmo(x, signature="default", as_dict=True)["elmo"]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())
        # return average of ELMo features
        return sess.run(tf.reduce_mean(embeddings,1))

#load data
data = np.load("data.npy", allow_pickle=True)
data = np.delete(data, (0), axis=0)
data = pd.DataFrame(data)

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
pickle_out = open("elmo_train_26042020.pickle","wb")
pickle.dump(elmo_train_new, pickle_out)
pickle_out.close()
# save elmo_test_new
pickle_out = open("elmo_test_26042020.pickle","wb")
pickle.dump(elmo_test_new, pickle_out)
pickle_out.close()

# load elmo_train_new
pickle_in = open("elmo_train_26042020.pickle", "rb")
elmo_train_new = pickle.load(pickle_in)
# load elmo_train_new
pickle_in = open("elmo_test_26042020.pickle", "rb")
elmo_test_new = pickle.load(pickle_in)

#split again for validation
xtrain, xvalid, ytrain, yvalid = train_test_split(elmo_train_new, train[4],  random_state=42, test_size=0.2)

#training logistinc regresion
print("training log reg...")
lreg = LogisticRegression()
lreg.fit(xtrain, ytrain)
print("done")

#predict
preds_valid = lreg.predict(xvalid)
print(f1_score(yvalid, preds_valid))

# make predictions on test set
preds_test = lreg.predict(elmo_test_new)
print(f1_score(test[4], preds_test))
