import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, recall_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

#converts words before to their respective sentiment(0,1,2) and learns on that vectors

def load_preprocess_data(limit=None):
    #load data
    data = np.load("data/data_v4.npy", allow_pickle=True)
    data = np.delete(data, (0), axis=0)
    data = data[data[:,4] != None]  #removing None sentiments
    data[:,4] = np.where(data[:,4] == '5', '4', data[:,4]) #change class 5->4
    data[:,4] = np.where(data[:,4] == '1', '2', data[:,4]) #change class 1->2
    data = pd.DataFrame(data)

    if limit != None:
        data = data[0:limit]

    #sentiment distribution
    print(data[4].value_counts(normalize = True))

    #train test split
    train, test = train_test_split(data, test_size=0.2, random_state=42)

    return train, test

def upsample_minority(df):
    df_majority = df.loc[df[4] == '3']
    df_minority_2 = df.loc[df[4] == '2']
    df_minority_4 = df.loc[df[4] == '4']
    df_minority = pd.concat([df_minority_2, df_minority_4])

    df_minority_upsampled = resample(df_minority,
                                     replace=True,  # sample with replacement
                                     n_samples=int(len(df_majority)/2),  # to match majority class
                                     random_state=42)  # reproducible results

    df_upsampled = pd.concat([df_majority, df_minority_upsampled])
    return df_upsampled

def fit_model(train, test):
    #prepare train/test in a way that lr understands it
    train_x = np.empty([3,])
    test_x = np.empty([3,])
    for row in train[6]: #vrstice
        zeros = sum([item.count(0) for item in row])
        ones  = sum([item.count(1) for item in row])
        twos  = sum([item.count(2) for item in row])
        new_row = [zeros, ones, twos]
        train_x = np.vstack([train_x, new_row])
    for row in test[6]: #vrstice
        zeros = sum([item.count(0) for item in row])
        ones  = sum([item.count(1) for item in row])
        twos  = sum([item.count(2) for item in row])
        new_row = [zeros, ones, twos]
        test_x = np.vstack([test_x, new_row])
    train_x = np.delete(train_x, (0), axis=0)
    test_x = np.delete(test_x, (0), axis=0)


    # training logistinc regresion
    print("training log reg...")
    lreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', dual=False, max_iter=1000, random_state=23)
    lreg.fit(train_x, train[4])
    print("done")

    # make predictions on test set
    preds_test = lreg.predict(test_x)
    print("F1 test: ", f1_score(test[4], preds_test, average='micro'))

    # test accuracy
    print("CA: ", accuracy_score(test[4], preds_test))

    # recall and ROC area
    print('Recall score: {}'.format(recall_score(test[4], preds_test, average='micro')))

    # confusion matrix
    labels = ['2', '3', '4']
    print(confusion_matrix(test[4], preds_test, labels))

    # training random forest
    print("training random forest...")
    rand_forest = RandomForestClassifier(n_estimators=50, random_state=69)
    rand_forest.fit(train_x, train[4])
    preds = rand_forest.predict(test_x)
    print("Random Forest Classification Score: ", rand_forest.score(test_x, test[4]))
    # confusion matrix
    labels = ['2', '3', '4']
    print(confusion_matrix(test[4], preds, labels))


#-------MAIN------
train, test = load_preprocess_data()

# upsample minority class in train dataset
# train = upsample_minority(train)
print(train[4].value_counts(normalize=True))

#fit the model
fit_model(train, test)