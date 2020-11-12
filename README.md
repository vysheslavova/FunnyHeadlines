# Data
**News**:
* [All the news](https://www.kaggle.com/snapcrack/all-the-news) (143,000)
* [10,700 articles from the front page of the Times](https://components.one/datasets/above-the-fold/) (10,700)

**Idioms**:
* [IBM Debater - Sentiment Lexicon of IDiomatic Expressions (SLIDE)](https://www.research.ibm.com/haifa/dept/vst/debating_data.shtml) (4,976)
* definition from [wiktionary.org](https://www.wiktionary.org)

**Quotes**:
* [Quotes Dataset](https://www.kaggle.com/akmittal/quotes-dataset) (9,415)

# Methods
* BM25 ([elasticsearch](https://www.elastic.co/elasticsearch/)): folder [elsearch](../blob/master/elsearch)
* [Sentence-BERT](https://arxiv.org/abs/1908.10084): [similarity.py](similarity.py)
* [USE](https://arxiv.org/abs/1803.11175): [similarity.py](/similarity.py)

# Articles 
* OpenIE and SpaCy dependencies, NERs, top unigrams: content.py

# Modification
**Idea**:
* we have an example of using the idiom (https://www.wiktionary.org or https://idioms.thefreedictionary.com)
* get pattern from example: get dependencies from example with words from idiom and type of these dependencies
  * idiom: bet the farm
  * example: "I'd be surprised if those two are still dating come Christmas, but I'm not betting the farm on a breakup just yet."
  * dependencies: [('I', 'betting', 'nsubj'), ('the farm', 'betting', 'dobj')]
* replace words in these dependencies which not in idiom by words from artcile (title/ners/...) (title dependencies: [title_dep.py](../blob/master/title_dep.py)
  * title: "'Everything I've Done Is 100 Percent Proper,' Trump Says of Russia Inquiry"
  * title dependencies: ('Everything', 'Is', 'nsubj', 1), ('I', 'Done', 'nsubj', 0), ('Russia', 'of', 'pobj', 3), ('Inquiry', 'of', 'pobj', 1)
  * result: ('Russia betting the farm', 3), ('Everything betting the farm', 1), ('Inquiry betting the farm', 1), ('I betting the farm', 0)
* [modification.ipynb](../blob/master/modification.ipynb)

# Labeled data
* [example](https://docs.google.com/spreadsheets/d/1XuIBp2oiyWjN5eZi0I6M1Wmv84cUKSLNiCyx6WG6Cqg/edit?usp=sharing)
* https://drive.google.com/drive/folders/1lgrUjz8IAc-coRRNZAJHhvUHG_640s04?usp=sharing
* for 20 articles top-5: search with bm25/sbert/use by title/lead, datasets idioms/quotes
