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
