import numpy as np

data = np.load("data/data_v4.npy", allow_pickle=True)

allWords = []
numOfLines = 0

for File_ID, Entity_ID, Entity_type, Entities, Sentiment, Words_before, Words_before_sentiments, Sentances in data[1:]:
    for words_before in Words_before:
        numOfLines += 1
        for word in words_before:
            word = word.lower()
            if word not in allWords:
                allWords.append(word)

print("All words done")

print("Num of words:", len(allWords))
print("Num of sent:", numOfLines)
print(allWords)

wordArray = []#np.zeros((len(allWords)+1, numOfLines))

dataLen = len(data)


for j, (File_ID, Entity_ID, Entity_type, Entities, Sentiment, Words_before, Words_before_sentiments, Sentances) in enumerate(data[1:]):

    if j > 100:
        break

    print(j, "/", dataLen)
    for words_before in Words_before:
        line = np.zeros(len(allWords)+1)

        for word1 in words_before:
            for i, word2 in enumerate(allWords):
                if word1 == word2:
                    line[i] = 1
                    continue

        line[-1] = Sentiment
        wordArray.append(line)


np.save('data/wordArray_v1_extra_small.npy', wordArray)
