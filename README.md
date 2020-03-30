# ONJ: Aspect-based sentiment analysis

## Introduction
Za seminarsko nalogo sva izbrala temo "Aspect-based sentiment analysis". Najprej sva pogledala podan nabor podatkov, ki jih boma uporabila ("Slovene corpus for aspect-based sentiment analysis - SentiCoref 1.0", url:https://www.clarin.si/repository/xmlui/handle/11356/1285) in pregledala njegovo strukturo in vsebino. Ugotovila sva, da so podatki že ustrzno anotirani in imajo označene entitete ter koreference. V tej seminarski nali se boma osredotočila na samo na del klasifikacije sentimenta posamznih entitet.

## Existing solutions
- Ding et. al.: Entity-Level Sentiment Analysis of Issue Comments. V članku je opisan pristop k klasifikaciji sentimenta v tri razrede za posamezne zaznane entitete v komentarjih na GitHub projektih. Na podatkih ki zajemajo 3000 komentajev so dosgli klasifikacijsko točnost 68%. Za klasifikacijo so uporabili različne metode nadzorovanega učenja. POstopek klasifikacije sentimenta je okvirno takšen: predobdelava(odstranitev nepomembnih besed, tokenizacija in stemming, vektorizacija z uporabo TF-IDF in Doc2Vec. Za samo učenje in klasifikacijo pa so uporabili Random Forest, Bagging, SVM, Naive-Bayes, Gradient Boosting. 

- Sweeney et. al.: Multi-entity sentiment analysis using entity-level feature extraction and
word embeddings approach. 

## Initial ideas
Predpostavila sva da so podatki o entitetah in koreferencah že podani in je torej ključna naloga klasifikacija sentimenta entitet.
Prvotna ideja, preden sva pogledala obsotojče rešitve je bila, da za posamezno entiteto iz besedila pridobima:
a) celotne stavke v katerih se entiteta pojavi oziroma na njih navezuje
b) n okoliških besed ob entiteti (enako besed pred in za entiteto: več besed pred, manj besed po entiteti, saj so mogoče besede pred entiteto bolj pomembne; ali pa ravno obratno).

Po tem bi za pridobljene stavke ali delov stavkov pridobila njihove sentimente, jih združila (s povprečenjem, ali uteževanjem). Sentimente bi pridobila s pomočjo različnih metod strojnega učenjastrojnega učenja (nevronske mreže, klasični pristopi).

Po pregledu obstoječih rešitev iz literature... #TODO

Problem, ki sva ga opazila pri podatkih je, da so podatki zelo neuravnoteženi. Podatkovna množica vsebuje 14,572 sentimentnih označb, od katerih je večina (~74%) nevtralnih, nekaj (~24%) negaitvih in pozitivnih in zelo malo (<1%) zelo negativnih oz. zelo pozitivnih. Ta problem mova poizkusila rešiti z utežitvijo (zelo) negativnih/pozitivnih sentimenov, ali napovedovanjem le nevtralnih, pozitivnih in negativnih sentimentov, pri čemer bi zelo negaitve/pozitivne združila z negativinimi/pozitivnimi.
