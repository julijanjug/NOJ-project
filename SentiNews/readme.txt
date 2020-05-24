Author: Jože Bučar, Faculty of Information Studies in Novo mesto (contact: joze.bucar@gmail.com)

Abstract:
Between 2 and 6 annotators independently sentiment annotated a stratified random sample of 10,427 documents from the Slovenian news portals 24ur, Dnevnik, Finance, Rtvslo, and Žurnal24. These portals contain political, business, economic and financial content. The texts were annotated using the five-level Lickert scale (1 – very negative, 2 – negative, 3 – neutral, 4 – positive, and 5 – very positive) on three levels of granularity, i.e. on the document, paragraph, and sentence level.

Annotation: 6 annotators, it took more than a year to manually annotate the sample; instructions to annotators: „Please specify the sentiment from the perspective of an average Slovenian web user. How did you feel after reading the news?“; five-level Lickert scale (1 – very negative, 2 – negative, 3 – neutral, 4 – positive, and 5 – very positive)

Levels of granularity: Document level, paragraph level, sentence level

Sentiment allocation: Negative (if average of scores ≤ 2.4); neutral (if average of scores is between 2.4 and 3.6); positive (average of annotated scores ≥ 3.6)

Short statistics: 10,427 documents; 89,999 paragraphs; 168,899 sentences; 3,261,327 words; 214,705 unique words

Keywords:
news corpus, sentiment classification, opinion mining

Web resources:
- Slovenian news texts with political, business, economic and financial content published between 1 September 2007 and 31 December 2013 from five Slovenian web media from five web media: www.24ur.com, www.dnevnik.si, www.finance.si, www.rtvslo.si, www.zurnal24.si

Type and size:
- .txt: Document level (22.66 MB), Paragraph level (23.21 MB), Sentence level (25.96 MB)

Encoding: UTF-8

Year: 2013-2014

Attributes:
nid - News ID [integer; from 1 to 10,427]
pid - Paragraph ID [integer; from 1 to 94]
sid - Sentence ID [integer; from 1 to 150]
main_url - Uniform Resource Locator (URL) of the resource (web medium) [string; www.24ur.com, www.dnevnik.si, www.finance.si, www.rtvslo.si, www.zurnal24.si]
url - URL of the news [string]
title - Title of the news [string]
keywords - Keywords of the news [string]
content - Content of the news (document) or its part (paragraph, sentence) [string]
date - Date of publishing the news [string; format: yyyy-mm-dd]
author - Author of the news [string]
Ann1 - Ann6 - Manual annotations from 6 annotators, they include scores on five-level Lickert scale (1 – very negative, 2 – negative, 3 – neutral, 4 – positive, and 5 – very positive) [integer; from 1 to 5]
avg_sentiment - Average of scores (Ann1 - Ann6) [float; from 1 to 5]
sd_sentiment - Standard deviation of scores (Ann1 - Ann6) [float; document level: from 0 to 1.225, paragraph level: from 0 to 2.828, sentence level: from 0 to 2.828]
sentiment - Sentiment allocation according to avg_sentiment score [string; negative, neutral, positive]

References:
Bučar, Jože, Janez Povh, Martin Žnidaršič (2016). Sentiment Classification of the Slovenian News Texts. Proceedings of the 9th International Conference on Computer Recognition Systems CORES 2015, Wrocław, Poland. Springer International Publishing, 2016, http://link.springer.com/chapter/10.1007/978-3-319-26227-7_73.
