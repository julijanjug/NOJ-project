# ONJ: Aspect-based sentiment analysis
Avtorja: Julijan Jug, Jaka Jenko

## Introduction
Za seminarsko nalogo sva izbrala temo "Aspect-based sentiment analysis". Najprej sva pogledala podan nabor podatkov, ki jih boma uporabila ("Slovene corpus for aspect-based sentiment analysis - SentiCoref 1.0", url:https://www.clarin.si/repository/xmlui/handle/11356/1285) in pregledala njegovo strukturo in vsebino. Ugotovila sva, da so podatki že ustrzno anotirani in imajo označene entitete ter koreference. V tej seminarski nali se boma osredotočila na samo na del klasifikacije sentimenta posamznih entitet.

## Initial ideas
Predpostavila sva da so podatki o entitetah in koreferencah že podani in je torej ključna naloga klasifikacija sentimenta entitet.
Prvotna ideja, preden sva pogledala obsotojče rešitve je bila, da za posamezno entiteto iz besedila pridobima:
a) celotne stavke v katerih se entiteta pojavi oziroma na njih navezuje
b) n okoliških besed ob entiteti (enako besed pred in za entiteto: več besed pred, manj besed po entiteti, saj so mogoče besede pred entiteto bolj pomembne; ali pa ravno obratno).

Za vectorizacijo bi uporabila word2vec in za klasificiralaposamezne stavke ali dele stavkov in določila njihov sentiment. Te klasifikacije pa nato nekako združila (s povprečenjem, ali uteževanjem). Sentimente bi pridobila s pomočjo različnih metod strojnega učenja (nevronske mreže, klasične metode).

Problem, ki sva ga opazila pri podatkih je, da so podatki zelo neuravnoteženi. Podatkovna množica vsebuje 14,572 sentimentnih označb, od katerih je večina (~74%) nevtralnih, nekaj (~24%) negaitvih in pozitivnih in zelo malo (<1%) zelo negativnih oz. zelo pozitivnih. Ta problem mova poizkusila rešiti z utežitvijo (zelo) negativnih/pozitivnih sentimenov, ali napovedovanjem le nevtralnih, pozitivnih in negativnih sentimentov, pri čemer bi zelo negaitve/pozitivne združila z negativinimi/pozitivnimi.


# Running the project  
All code for data preprocessing and model building is located in folder CodeToTest. The project requires Python 3.

## Required libraries:
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
