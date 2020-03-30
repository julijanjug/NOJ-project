# ONJ: Aspect-based sentiment analysis

## Introduction
Za seminarsko nalogo sva se odločila izbrati temo "Aspect-based sentiment analysis". Najprej sva pogledala podan nabor podatkov ("Slovene corpus for aspect-based sentiment analysis - SentiCoref 1.0", url:https://www.clarin.si/repository/xmlui/handle/11356/1285) in si ga razložila. Ugotovila sva, da so podatki anotirani in združeni v entitete, katere imajo ob azdnji pojavitvi določen sentiment.

## Existing solutions


## Initial ideas
Prvotna ideja, preden sva pogledala obsotojče rešitve je bila, da za posamezno entiteto iz besedila dobima:
a) celotne stavke v katerih se entiteta pojavi ali
b) n okoliških besed ob entiteti (enako besed pred in za entiteto: več besed pred, manj besed po entiteti, saj so mogoče besede pred entiteto bolj pomembne; ali pa ravno obratno).

Po tem bi za pridobljene stavke ali delov stavkov pridobila njihove sentimente, jih združila (s povprečenjem, ali uteževanjem). Sentimente bi pridobila s pomočjo strojnega učenja (nevronske mreže).

Problem, ki sva ga opazila pri podatkih je, da so podatki zelo neuravnoteženi. Podatkovna množica vsebuje 14,572 sentimentnih označb, od katerih je večina (~74%) nevtralnih, nekaj (~24%) negaitvih in pozitivnih in zelo malo (<1%) zelo negativnih oz. zelo pozitivnih. Ta problem mova poizkusila rešiti z utežitvijo (zelo) negativnih/pozitivnih sentimenov, ali napovedovanjem le nevtralnih, pozitivnih in negativnih sentimentov, pri čemer bi zelo negaitve/pozitivne združila z negativinimi/pozitivnimi.
