import spacy
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


def title_chunks(title):
    """
    input:
        title - title of article
    output:
        SpaCy noun_chunks for title
    """
    nlp = spacy.load("en_core_web_sm")
    name = []
    doc = nlp(title)
    for chunk in doc.noun_chunks:
        name.append((chunk.text, chunk.root.head.text, chunk.root.dep_))
    return name


def get_score(data, title, i):
    """
    input:
        data - articles pd.DataFrame()
        title - title of article
        i - index of article in data
    output:
        score for dependencies: counter by all stems

    """
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    vocabulary = []
    for item in data['lead']: # or content
        vocab = {}
        item = stemmer.stem(item)
        for word in word_tokenize(item):
            if word not in stop_words:
                try:
                    vocab[word] += 1
                except KeyError:
                    vocab[word] = 1
        vocabulary.append(vocab)

    all_stems = []
    dependencies = title_chunks(title)
    for dep in dependencies:
        stems = []
        for word in dep[0].split():
            st = stemmer.stem(word.lower())
            if st[-1] == "'":
                st = st[:-1]
            stems.append((st, lemmatizer.lemmatize(word.lower())))
        all_stems.append(stems)

    score = []
    keys = list(vocabulary[i].keys())
    for elem in all_stems:
        n = 0
        for stem, lemm in elem:
            for key in keys:
                if re.search(stem, key) and len(stem) >= 4 or re.search(lemm, key) and len(
                        lemm) >= 4 or stem == key or lemm == key:
                    n += vocabulary[i][key]
        score.append(n)

    final = []
    for i, dep in enumerate(dependencies):
        final.append((dep[0], dep[1], dep[2], score[i]))

    return final

if __name__ == '__main__':
    articles = pd.read_csv('articles_20.csv')
    dependencies = []
    for i, item in enumerate(articles['title']):
        dependencies.append(get_score(articles, item, i))