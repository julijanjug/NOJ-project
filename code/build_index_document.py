import numpy as np
import pandas as pd
from nltk.corpus import state_union, stopwords
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer
from nltk.tokenize import word_tokenize
import xx_ent_wiki_sm
nlp = xx_ent_wiki_sm.load()
from statistics import mean, mode
import sqlite3
import re
from collections import Counter


def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None
cnx = sqlite3.connect(':memory:')
cnx.create_function("REGEXP", 2, regexp)
cursor = cnx.cursor()
lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
stop_words = set(stopwords.words('slovene'))


def load_preprocess_data(limit=None):
    #load data
    data = np.load("../data/data_v4.npy", allow_pickle=True)
    data = np.delete(data, (0), axis=0)
    data = data[data[:,4] != None]  #removing None sentiments
    data = pd.DataFrame(data)

    if limit != None:
        data = data[0:limit]

    #text preprocesing
    # convert text to lowercase
    data[3] = data[3].apply(lambda x: [s.lower() for s in x])
    # remove numbers
    data[3] = data[3].apply(lambda x: [s.replace("[0-9]", " ") for s in x])
    # remove whitespaces
    data[3] = data[3].apply(lambda x: [' '.join(s.split()) for s in x])
    #lematize the entities
    data[3] = data[3].apply(lambda x: [lemmatizer.lemmatize(s) for s in x])
    #remove stopwords if present
    data[3] = data[3].apply(lambda x: [s for s in x if s not in stop_words])
    #transform into set
    data[3] = data[3].apply(lambda x: set(x))

    entities_set = set()
    for entities in data[3]:
        entities_set = entities_set.union(entities)

    # transform into string
    data[3] = data[3].apply(lambda x: ','+ ','.join(x) +',')

    return data.iloc[:, 0:4], entities_set

#load all entities
entitete, entitete_union = load_preprocess_data()

#entitete to sql table
entitete.columns = ["doc_id", "ent_id", "ent_type", "ent"]
entitete.to_sql(name='entitete', con=cnx)

avg_sen_dict = dict()
sd_sen_dict = dict()
sen_dict = dict()

file = "../SentiNews/SentiNews_document-level.txt"
with open(file, encoding="utf-8") as f:
    header = f.readline()

    for line in f:
        splitted = line.split("\t")
        print("Document: ",splitted[0])
        avg_sentiment = splitted[14]
        sd_sentiment = splitted[15]
        sentiment = splitted[16]
        content = splitted[5]

        doc = nlp(content)
        doc_ents = [word_tokenize(X.text) for X in doc.ents if X.label_ in {'PER', 'ORG', 'LOC'}]

        tokens = set()
        for ent in doc_ents:
            ent = [lemmatizer.lemmatize(e.lower()) for e in ent]
            ent = [e for e in ent if e not in stop_words and e in entitete_union]
            tokens = tokens.union(set(ent))

        # test = pd.read_sql("""select * from entitete""", cnx)
        reg = "'," + ",|,".join(tokens) + ",'"
        query = "select * from entitete where entitete.ent REGEXP ?"
        test = cursor.execute(query, [reg])
        data = cursor.fetchall()

        for row in data:
            key = row[1]+"-"+row[2]
            if key not in avg_sen_dict:
                avg_sen_dict[key] = []
                sd_sen_dict[key] = []
                sen_dict[key] = []
            avg_sen_dict[key].append(avg_sentiment)
            sd_sen_dict[key].append(sd_sentiment)
            sen_dict[key].append(sentiment)

# rezultate damo v en ndarray
result = np.empty([4,])
for key, value in avg_sen_dict.items():
    avg_sen    = mean([float(item) for item in value])
    avg_st_sen = mean([float(item) for item in sd_sen_dict[key]])
    tmp = Counter(sen_dict[key])
    mode_sen   = tmp.most_common(1)[0][0].replace("\n", "")
    new_row    = [key, avg_sen, avg_st_sen, mode_sen]
    result = np.vstack([result, new_row])

#stolpci: docid+entid, avg_sentiment, avg_sd_sentiment, mode_sentiment
np.save("../SentiNews/docLevel_sentiment_v1.npy", result)

# test = np.load("docLevel_sentiment_v1.npy", allow_pickle=True)
# print(test)