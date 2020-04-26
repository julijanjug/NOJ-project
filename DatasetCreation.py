import glob
import numpy as np
import sys
import nltk
from nltk.corpus import state_union, stopwords
from nltk.tokenize import PunktSentenceTokenizer, sent_tokenize
import lemmagen.lemmatizer
from lemmagen.lemmatizer import Lemmatizer


# stop wordi stran
# lematizacija
# stavki samo do pike

NUM_WORDS_BACK = 5
STOP_WORDS = [".", "!", "?", "..."]
LOCILA = [".", "!", "?", "...", ",", ";", ":", "-", "–", "\"", "(", ")", "\\", "/", "[", "]", "{", "}", "@", "”", "…", "•"]


def IsEnitity(line):
    if line.split("\t")[6] != "_":
        return True
    return False

def GetWord(line):
    return line.split("\t")[2]

def GetEntittyID(line):
    return line.split("\t")[6].split("[")[-1].split("]")[0]

def GetEntityType(line):
    type = line.split("\t")[3]

    if type == "_":
        return False

    if "[" in type:
        return type.split("[")[0]

    return type

def GetSentimentIfHasIt(line):
    sentiment = line.split("\t")[4]

    if sentiment != "_":
        return sentiment.split(" ")[0]

    return False

def GetWordFirstCharacterLocation(line):
    return int(line.split("\t")[1].split("-")[0])

def GetCorrectSentance(allSentances, characterLocation):
    for sentance in allSentances:
        if sentance[0][0] <= characterLocation <= sentance[0][1]:
            return sentance[1]

lemmatizer = Lemmatizer(dictionary=lemmagen.DICTIONARY_SLOVENE)
stop_words = set(stopwords.words('slovene'))

file1 = open("data/data_v3.txt", "w")
allData = np.array(["File_ID", "Entity_ID", "Entity_type", "Entities", "Sentiment", "Words_before", "Sentances"])
file1.write(np.array2string(allData) + "\n")


for file in glob.glob("SentiCoref_1.0/*.tsv"):
    fileName = file.split("\\")[-1].split(".")[0]
    print(fileName)

    with open(file, encoding="utf-8") as f:

        start = 0
        readData = False

        all_sentances = []

        workingEntityId = None
        wordsBack = []

        entitiesInFile = dict()

        for line in f:
            if not readData:
                if "#Text" in line:
                    for sent in sent_tokenize(line.replace("#Text=", "").replace(" .", "."), language='slovene'):
                        add = sent.count(".")
                        end = start + len(sent) + add
                        #print(start, end, sent)

                        filtered_sentence = []

                        for w in sent.split(" "):
                            if w not in stop_words:
                                w = lemmatizer.lemmatize(w)
                                filtered_sentence.append(w)

                        all_sentances.append(([start, end], ' '.join(filtered_sentence)))

                        start = end + 1

                        #print('---------')
                    readData = True
            else:
                line = line.replace('\n', ' ').replace('\r', '')
                word = GetWord(line)

                if word in LOCILA:
                    continue

                if IsEnitity(line):
                    currentEntityId = GetEntittyID(line)

                    # If entity already seen
                    if currentEntityId in entitiesInFile.keys():
                        entitiesInFile[currentEntityId][3].add(word)

                        if workingEntityId != currentEntityId:
                            workingEntityId = currentEntityId

                            if entitiesInFile[currentEntityId][5] is None:
                                entitiesInFile[currentEntityId][5] = wordsBack.copy()
                            else:
                                entitiesInFile[currentEntityId][5].append(wordsBack.copy())

                            entitiesInFile[currentEntityId][6].append(GetCorrectSentance(all_sentances, GetWordFirstCharacterLocation(line)))


                        sentiment = GetSentimentIfHasIt(line)
                        if sentiment:
                            entitiesInFile[currentEntityId][4] = sentiment

                        if entitiesInFile[currentEntityId][2] is None:
                            entityType = GetEntityType(line)
                            if entityType:
                                entitiesInFile[currentEntityId][2] = entityType

                    # If new entity
                    else:
                        # filename, entity id, entity words, sentiment, words before entity
                        entitiesInFile[currentEntityId] = [fileName, currentEntityId, None, set(), None, None, [GetCorrectSentance(all_sentances, GetWordFirstCharacterLocation(line))]]
                        entitiesInFile[currentEntityId][3].add(word)

                        if len(wordsBack) > 0:
                            entitiesInFile[currentEntityId][5] = wordsBack.copy()

                        sentiment = GetSentimentIfHasIt(line)
                        if sentiment:
                            entitiesInFile[currentEntityId][4] = sentiment

                        entityType = GetEntityType(line)
                        if entityType:
                            entitiesInFile[currentEntityId][2] = entityType
                else:
                    workingEntityId = None

                if word in STOP_WORDS:
                    wordsBack = []
                else:
                    wordsBack.append(word)

                    if len(wordsBack) > NUM_WORDS_BACK:
                        wordsBack.pop(0)

        for key in entitiesInFile.keys():
            npArray = np.array([entitiesInFile[key][0], entitiesInFile[key][1], entitiesInFile[key][2], list(entitiesInFile[key][3]), entitiesInFile[key][4], entitiesInFile[key][5], entitiesInFile[key][6]])
            allData = np.vstack((allData, npArray))
            #print([entitiesInFile[key][0], entitiesInFile[key][1], list(entitiesInFile[key][2])])
            file1.write(np.array2string(npArray, separator=',').replace('\n', '') + "\n")

        #print()
        #print(allData.shape)

np.save('data/data_v3.npy', allData)

file1.close()