import pandas as pd
import numpy as np
from tqdm import tqdm
from openie import StanfordOpenIE
from collections import Counter, OrderedDict
import spacy
import pandas as pd
from spacy.lang.en.stop_words import STOP_WORDS as stop_words
import stanza
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
stanza.download('en')

def openie_subj(data):
    """
    input:
        data - column of pd.DataFrame() ('content'/'lead'/...)
    output:
        subjects from OpenIE
    """
    dependences = []
    with StanfordOpenIE() as client:
        for item in tqdm(data, total=len(data)):
            dependency = []
            relations = client.annotate(item)
            for r in relations:
                words = tuple(r.values())
                dependency.append(words)
            objects = list(map(lambda x: x[0], dependency))
            c = Counter(objects)
            common = OrderedDict(
                {key: val for key, val in sorted(c.items(), key=lambda item: item[1], reverse=True) if val > 1})
            dependences.append(common)
    return dependences


def spacy_subj(data):
    """
    input:
        data - column of pd.DataFrame() ('content'/'lead'/...)
    output:
        subjects from SpaCy noun_chunks
    """
    nlp = spacy.load("en_core_web_sm")
    names = []
    for item in tqdm(data):
        name = []
        doc = nlp(item)
        for chunk in doc.noun_chunks:
            name.append(chunk.text)
        c = Counter(name)
        common = OrderedDict({key: val for key, val in sorted(c.items(), key=lambda item: item[1], reverse=True)
                              if key.lower() not in stop_words})
        names.append(common)
    return names

def ners(data):
    """
    input:
        data - column of pd.DataFrame() ('content'/'lead'/...)
    output:
        NERs with count
    """
    nlp = stanza.Pipeline('en')
    ners = []
    for item in tqdm(data):
        doc = nlp(item)
        ner = [(ent.text, ent.type) for ent in doc.entities if ent.type == 'ORG' or ent.type == 'PERSON']
        c = Counter([ent[0].lower() for ent in ner])
        common = OrderedDict({key: val for key, val in sorted(c.items(), key=lambda item: item[1], reverse=True) if
                              key not in stop_words})
        ners.append(common)
    return ners

def top_ngrams(data):
    my_stop_words = text.ENGLISH_STOP_WORDS.union(["dont"], ["didnt"], ["shed"], ["wouldnt"])
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 1), stop_words=my_stop_words, max_df=0.95)
    corpus = data.values
    vectors = vectorizer.fit_transform(corpus)
    vocabulary = vectorizer.get_feature_names()
    unigrams = []
    # bigrams = []
    for vect in vectors:
        w = {'unigrams': []}
        # w = {'unigrams': [], 'bigrams': []}
        vect = np.array(vect.todense()).squeeze(0)
        for idx in reversed(np.argsort(vect)):
            word = vocabulary[idx]
            if len(word.split()) == 1 and len(w['unigrams']) < 5:
                w['unigrams'] += [word]
            # elif len(word.split()) == 2 and len(w['bigrams']) < 5:
            #     w['bigrams'] += [word]
            # elif len(w['unigrams']) >= 5 and len(w['bigrams']) >= 5:
            elif len(w['unigrams']) >= 5:
                break
        unigrams.append(', '.join(w['unigrams']))
        # bigrams.append(', '.join(w['bigrams']))
    return unigrams

if __name__ == '__main__':
    articles = pd.read_csv('articles_20.csv')
    article = articles.iloc[0]
    article['subjects'] = openie_subj([article['lead']])
    article['chunks'] = spacy_subj([article['lead']])
    article['ners'] = ners([article['lead']])
    articles['unigrams'] = top_ngrams(articles['lead'])