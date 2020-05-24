import tensorflow as tf
import os
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import (TFBertForSequenceClassification, BertTokenizer)
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
print(f"Tensorflow version: {tf.__version__}")

# Restrict TensorFlow to only allocate 4GBs of memory on the first GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(f"The system contains '{len(gpus)}' Physical GPUs and '{len(logical_gpus)}' Logical GPUs")
  except RuntimeError as e:
    print(e)
else:
    print(f"Your system does not contain a GPU that could be used by Tensorflow!")

#read data
def load_preprocess_data(limit=None):
    #load data
    data = np.load("../data/data_v4.npy", allow_pickle=True)
    data = np.delete(data, (0), axis=0)
    data = data[data[:,4] != None]  #removing None sentiments
    # data[:,4] = np.where(data[:,4] == '5', '1', data[:,4]) #change class 5->4
    # data[:,4] = np.where(data[:,4] == '1', '1', data[:,4]) #change class 1->2
    data[:,4] = np.where(data[:,4] == '1', 1, data[:,4]) #change class 1->1
    data[:,4] = np.where(data[:,4] == '2', 1, data[:,4]) #change class 2->1
    data[:,4] = np.where(data[:,4] == '3', 0, data[:,4]) #change class 1->0
    data[:,4] = np.where(data[:,4] == '4', 1, data[:,4]) #change class 4->1
    data[:,4] = np.where(data[:,4] == '5', 1, data[:,4]) #change class 5->1
    data = pd.DataFrame(data)

    if limit != None:
        data = data[0:limit]

    #sentiment distribution
    print(data[4].value_counts(normalize = True))

    #text preprocesing
    # convert text to lowercase
    # data[7] = data[7].apply(lambda x: [s.lower() for s in x])
    # remove numbers
    # data[7] = data[7].apply(lambda x: [s.replace("[0-9]", " ") for s in x])
    # remove whitespaces
    # data[5] = data[5].apply(lambda x: [' '.join(s.split()) for s in x])

    #train test split
    # train, test = train_test_split(data, test_size=0.2, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(data[7], data[4], test_size=0.2, random_state=0)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

    return X_train, y_train, X_test, y_test, X_val, y_val

#make test, train, val sets
X_train, y_train, X_test, y_test, X_val, y_val = load_preprocess_data(500)

#load the bert pretrained moed and tokenizer
bert_model = TFBertForSequenceClassification.from_pretrained("bert-base-cased")
bert_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

#prepare input for tokenizer
pad_token = 0
pad_token_segment_id = 0
max_length = 128

def convert_to_input(reviews):
    input_ids, attention_masks, token_type_ids = [], [], []

    for x in tqdm(reviews, position=0, leave=True):
        inputs = bert_tokenizer.encode_plus(x, add_special_tokens=True, max_length=max_length)

        i, t = inputs["input_ids"], inputs["token_type_ids"]
        m = [1] * len(i)

        padding_length = max_length - len(i)

        i = i + ([pad_token] * padding_length)
        m = m + ([0] * padding_length)
        t = t + ([pad_token_segment_id] * padding_length)

        input_ids.append(i)
        attention_masks.append(m)
        token_type_ids.append(t)

    return [np.asarray(input_ids),
            np.asarray(attention_masks),
            np.asarray(token_type_ids)]

X_test_input=convert_to_input(X_test)
X_train_input=convert_to_input(X_train)
X_val_input=convert_to_input(X_val)

#transform data to tensorflow format
def example_to_features(input_ids,attention_masks,token_type_ids,y):
  return {"input_ids": input_ids,
          "attention_mask": attention_masks,
          "token_type_ids": token_type_ids},y

train_ds = tf.data.Dataset.from_tensor_slices((X_train_input[0], X_train_input[1],X_train_input[2],y_train)).map(example_to_features).shuffle(100).batch(24).repeat(5)
val_ds = tf.data.Dataset.from_tensor_slices((X_val_input[0], X_val_input[1],X_val_input[2],y_val)).map(example_to_features).batch(24)
test_ds = tf.data.Dataset.from_tensor_slices((X_test_input[0], X_test_input[1],X_test_input[2],y_test)).map(example_to_features).batch(24)

#prepare the model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
bert_model.compile(optimizer=optimizer, loss=loss, metrics=[metric])

print("Fine-tuning BERT on dataset")
bert_history = bert_model.fit(train_ds, epochs=3, validation_data=val_ds)

#results aftera a few epochs
results_true = test_ds.unbatch()
results_true = np.asarray([element[1].numpy() for element in results_true])
print(results_true)

#predictions for the test
results = bert_model.predict(test_ds)
print("Model predictions:")
# for i in range(0,15):
#     print(f"\t {results[0+i*128]}")

results_predicted = np.argmax(results, axis=1)

#F-1 sore
print(f"F1 score: {f1_score(results_true, results_predicted)}")
print(f"Accuracy score: {accuracy_score(results_true, results_predicted)}")