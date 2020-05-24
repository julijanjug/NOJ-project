# ONJ: Aspect-based sentiment analysis
Authors: Julijan Jug, Jaka Jenko

## Introduction
The project analyses different classification models and feature sets for aspect-based sentiment prediction. Previous research has shown that aspect-based sentiment prediction is a very hard problem, which is magnified due to the peculiarities of the Slovene language.
The experiments were conducted using data from SentiCoref 1.0. The data set was prepossessed by the extraction of entities, corresponding sentiments, and the surrounding words. the words were lemmatized and cleared of stop words. Additional data with more examples of how the entities are used was also added with the help of SentiNews corpus and Slovene sentiment lexicon.
Baseline results show a slight improvement over the majority class with 0.65 classification accuracy and the F-score of 0.64. Better results were obtained with the use of extended data (SentiNews) and the use of the ELMo training model with 0.75 classification accuracy and the F-score of 0.70.
Future work would use the combination of multiple used approaches and classification of first into two classes neutral and non-neutral and then categorize the non-neutral class in to positive and negative sentiments to improve the classification model performance.

Žitnik, Slavko, 2019, Slovene corpus for aspect-based sentiment analysis - SentiCoref 1.0, Slovenian language resource repository CLARIN.SI, http://hdl.handle.net/11356/1285.  
Kadunc, Klemen and Robnik-Šikonja, Marko, 2017, Slovene sentiment lexicon KSS 1.1, Slovenian language resource repository CLARIN.SI, http://hdl.handle.net/11356/1097.  
Bučar, Jože, 2017, Manually sentiment annotated Slovenian news corpus SentiNews 1.0, Slovenian language resource repository CLARIN.SI, http://hdl.handle.net/11356/1110.  


# Running the project  
All code for data preprocessing and model building is located in folder /code. The project requires Python 3.

## Required libraries:
Libraries can be installed using pip or conda comands.  
-numpy  
-nltk  
-lemmagen  
-sklearn  
-tensorflow v1  
-pickle  
-pandas  
-sqlite3  
-xx_ent_wiki_sm  

## Scripts:
- **DatasetCreation.py**: In it we create our initial preprocessed data set from SentiCoref 1.0 dataset. There we parse
Entities and its Sentiment and combine them with 5 words that are found before the entity and sentance in with the word
is found. Stop words are then removed and left words are lemmatized. For each of the 5 words found we also set its
sentimets  with the help of files found in "Negative_positive" folder.  
- **WordArrayCreation.py**: Uses preprocessed data set to create a huge array with all rows corresponding to 5 words found
before entity and columns corresponding to words. The array is then filled with 1s if the word is one of the 5 words
before the entity and 0s if it's not. The last column is the sentiment for the entity. (cirrently is set to only create
an array of 100 entities)  
- **wordArray_v1_extra_small.npy**: File created with WordArrayCreation.py  (we couldn't upload the 
full dataset because of Github limitations)  
- **WordArrayLearning.py**: Results for wordArray_v1.npy data set
- **build_index_document.py, build_index_paragraph.py, build_index_sentence.py**: Python scripts for building indexes using data from SentiNews corpus on document, paragraph and sentence level. The output files are saved in SentiNews directory.
- **model_all_2class.py, model_all_3class.py**: Scripts for building the train and test data sets using SentiNews, ELMo and KSS features. The output is classification accuracy and F-score for classifing 2 or 3 target classes using majority classifier, RandomForest and LogisticRegression models.
- **elmo_model_3class.py**: Script for building the train and test data sets using only ELMo embeddings. The output is classification accuracy and F-score for classifing 2 or 3 target classes using majority classifier, RandomForest and LogisticRegression models.
