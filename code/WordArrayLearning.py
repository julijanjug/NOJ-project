import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import time
import multiprocessing
import sys
import pandas as pd

from sklearn.utils import resample


def valueCounts(data):
    values = dict()

    for d in data:
        if d not in values.keys():
            values[d] = 1
        else:
            values[d] += 1



    for key, value in values.items():
        print(key, ":", value/len(data))



KNN_model = KNeighborsClassifier(n_neighbors=3)
NB_model = GaussianNB()
DT_model = DecisionTreeClassifier()

RF_model = RandomForestClassifier(max_depth=20, random_state=0)
RF_model2 = RandomForestClassifier(max_depth=25, random_state=0)
RF_model3 = RandomForestClassifier(max_depth=30, random_state=0)

NN_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(578, 100, 50), random_state=1)
NN2_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(578, 100, 100, 50), random_state=1)
NN3_model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(578, 100, 100, 100, 50), random_state=1)


data = np.load("data/wordArray_v1.npy", allow_pickle=True)
print("Data loaded")
print("Data length:", data.shape)


data = np.array([d for d in data if not np.isnan(d[-1])])

data[:,-1] = np.where(data[:,-1] == 5, 4, data[:,-1]) #change class 5->4
data[:,-1] = np.where(data[:,-1] == 1, 2, data[:,-1]) #change class 1->2

#0 - neutral
#1 else
#data[:,-1] = np.where(data[:,-1] == 3, 0, data[:,-1]) #change class 3->0
#data[:,-1] = np.where(data[:,-1] == 4, 1, data[:,-1]) #change class 4->1
#data[:,-1] = np.where(data[:,-1] == 2, 1, data[:,-1]) #change class 2->1

#columns_to_keep = np.append(np.sum(data[:, :-1], axis = 0) > 10, [True])
#data = data[:, columns_to_keep]

#rows_to_keep = np.sum(data[:, :-1], axis = 1) > 2
#data = data[rows_to_keep, :]

print("Data fixed")
print("Data length:", data.shape)

#print(data[-1])
#data = data[0:518]

#print(data[517])
#for i in data[517]:
#    print(i)


xTrain, xTest, yTrain, yTest = train_test_split(data[:, :-1], data[:,-1], test_size=0.20, random_state=0)
data = None
print("Data splited")

print(valueCounts(yTrain))
print(valueCounts(yTest))

#print(xTrain[0], len(xTrain[0]))
#print(yTrain[0])

#for i in xTrain[0]:
    #print(i)


start_time = time.time()
RF_model3.fit(xTrain, yTrain)
print("RF3 done")
RF_prediction3 = RF_model3.predict(xTest)
print("RF3 predict done")
print("--- %s seconds ---" % (time.time() - start_time))
print("RF3")
print(accuracy_score(RF_prediction3, yTest))
#print(classification_report(RF_prediction3, yTest))
print("F1 test: ", f1_score(yTest, RF_prediction3, average='weighted'))
print()


# training logistinc regresion
print("training log reg...")
lreg = LogisticRegression(solver='lbfgs', dual=False, max_iter=10000, random_state=23)
lreg.fit(xTrain, yTrain)
print("done")

# log reg scores
preds_test = lreg.predict(xTest)
print("F1 test: ", f1_score(yTest, preds_test, average='weighted'))
print("CA: ", accuracy_score(yTest, preds_test))
print('Recall score: {}'.format(recall_score(yTest, preds_test, average='weighted')))
#labels = ['1', '0', '2']
#print(confusion_matrix(yTest, preds_test, labels))

# training random forest
print("training random forest...")
rand_forest = RandomForestClassifier(n_estimators=10, random_state=69)
rand_forest.fit(xTrain, yTrain)
preds = rand_forest.predict(xTest)
print("Random Forest CA: ", rand_forest.score(xTest, yTest))
print("F1 test: ", f1_score(yTest, preds, average='weighted'))
#print(confusion_matrix(yTest, preds, labels))

'''
start_time = time.time()
KNN_model.fit(xTrain, yTrain)
print("KNN done")
KNN_prediction = KNN_model.predict(xTest)
print("KNN predict done")
print("--- %s seconds ---" % (time.time() - start_time))
print("KNN")
print(accuracy_score(KNN_prediction, yTest))
print(classification_report(KNN_prediction, yTest))
print()



start_time = time.time()
NB_model.fit(xTrain, yTrain)
print("NB done")
NB_prediction = NB_model.predict(xTest)
print("NB predict done")
print("--- %s seconds ---" % (time.time() - start_time))
print("NB")
print(accuracy_score(NB_prediction, yTest))
print(classification_report(NB_prediction, yTest))
print()



start_time = time.time()
DT_model.fit(xTrain, yTrain)
print("DT done")
DT_prediction = DT_model.predict(xTest)
print("DT predict done")
print("--- %s seconds ---" % (time.time() - start_time))
print("DT")
print(accuracy_score(DT_prediction, yTest))
print(classification_report(DT_prediction, yTest))
print()



start_time = time.time()
RF_model.fit(xTrain, yTrain)
print("RF done")
RF_prediction = RF_model.predict(xTest)
print("RF predict done")
print("--- %s seconds ---" % (time.time() - start_time))
print("RF")
print(accuracy_score(RF_prediction, yTest))
print(classification_report(RF_prediction, yTest))
print()



start_time = time.time()
RF_model2.fit(xTrain, yTrain)
print("RF2 done")
RF_prediction2 = RF_model2.predict(xTest)
print("RF2 predict done")
print("--- %s seconds ---" % (time.time() - start_time))
print("RF2")
print(accuracy_score(RF_prediction2, yTest))
print(classification_report(RF_prediction2, yTest))
print()



start_time = time.time()
RF_model3.fit(xTrain, yTrain)
print("RF3 done")
RF_prediction3 = RF_model3.predict(xTest)
print("RF3 predict done")
print("--- %s seconds ---" % (time.time() - start_time))
print("RF3")
print(accuracy_score(RF_prediction3, yTest))
print(classification_report(RF_prediction3, yTest))
print()



start_time = time.time()
NN_model.fit(xTrain, yTrain)
print("NN done")
NN_prediction = NN_model.predict(xTest)
print("NN predict done")
print("--- %s seconds ---" % (time.time() - start_time))
print("NN")
print(accuracy_score(NN_prediction, yTest))
print(classification_report(NN_prediction, yTest))
print()

start_time = time.time()
NN2_model.fit(xTrain, yTrain)
print("NN2 done")
NN2_prediction = NN2_model.predict(xTest)
print("NN2 predict done")
print("--- %s seconds ---" % (time.time() - start_time))
print("NN2")
print(accuracy_score(NN2_prediction, yTest))
print(classification_report(NN2_prediction, yTest))
print()

start_time = time.time()
NN3_model.fit(xTrain, yTrain)
print("NN3 done")
NN3_prediction = NN3_model.predict(xTest)
print("NN3 predict done")
print("--- %s seconds ---" % (time.time() - start_time))
print("NN3")
print(accuracy_score(NN3_prediction, yTest))
print(classification_report(NN3_prediction, yTest))
print()
'''