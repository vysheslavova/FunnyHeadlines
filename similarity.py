import pandas as pd
import numpy as np
import nltk
import re
import torch
import tensorflow as tf
import scipy
from tqdm import tqdm
import tensorflow_hub as hub
from sentence_transformers import SentenceTransformer
from sklearn import preprocessing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def filters(tpl, dataset, nums=5, min_verbs=0, min_len=2, max_len=10):
    """
    input:
        tpl - tuple (index, distance) from cosine distance
        dataset - dataset with candidates (idioms)
        nums - top-k
        min_verbs - min number of verbs in candidate
        min_len - min length of candidate
        max_len - max lenght of candodate
    output:
        filtered tpl
    """
    result = []
    for idx, dist in tpl:
        verbs = 0
        title = dataset['title'][idx]
        text = re.sub(r'[^\w\s]', '', title)
        words = nltk.tokenize.word_tokenize(text)[:-1]
        for word in words:
            tag = nltk.pos_tag([word])[0][1]
            if re.match('VB*', tag) is not None:
                verbs += 1

        if verbs >= min_verbs and max_len >= len(words) >= min_len:
            result.append((idx, dist))
        if len(result) >= nums:
            return result
    return result

def weighted(tpl, scores):
    """
    input:
        tpl - tuple (index, distance) from cosine distance
        scores - score from data (popularity)
    output:
        reranked tpl
    """
    distances = [y for x, y in tpl]
    scaler = preprocessing.MinMaxScaler((0, 10))
    distances = scaler.fit_transform((1 - np.array(distances)).reshape(-1, 1))[:, 0]
    tpl = [(x[0], y) for x, y in zip(tpl, distances)]
    result = []
    for idx, dist in tpl:
        score = scores[idx]
        result.append((idx, dist * 0.8 + 0.2 * score))
    return result

def search(method, articles, candidates, top=1):
    """
    input:
        method - method for search: 'sbert' or 'use'
        articles - dataset with articles
        candidates - dataset with candidates (idioms/movies/...)
        top - top-k for each query
    output:
        pd.DataFrame() with top candidates
    """
    assert method == 'sbert' or method == 'use'

    candidates.dropna(subset=['definition'], inplace=True)
    candidates.reset_index(drop=True, inplace=True)

    if method == 'sbert':
        model = SentenceTransformer('bert-large-nli-stsb-mean-tokens')
        model.to(device)
        sentence_embeddings = model.encode(candidates['definition'].iloc[:10])
        def embed(input):
            return model.encode(input)

    if method == 'use':
        module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
        model = hub.load(module_url)
        def embed(input):
            return model(input).numpy()
        candidates_embeddings = []
        for x in candidates['definition'].iloc[:10]:
            candidates_embeddings.append(embed([x]))
        sentence_embeddings = np.concatenate(candidates_embeddings, axis=0)


    result = []
    for i, query in tqdm(articles.iloc[:5].iterrows(), total=len(articles)):
        queries = [query['title']]  # search by title

        # for content search
        # queries = [query['title'] + '. ' + ' '.join(nltk.sent_tokenize(query['content'])[:2])]
        # queries = [query['title'] + query['lead']]

        query_embeddings = embed(queries)

        distances = scipy.spatial.distance.cdist(query_embeddings, sentence_embeddings, "cosine")[0]

        results = list(zip(range(len(distances)), distances))
        # results = weighted(results, idioms['score'])
        results = sorted(results, key=lambda x: x[1], reverse=False)
        # results = filters(results, number_top_matches, 1)

        for idx, distance in results[0:top]:
            result.append({'article': query['title'], 'lead': query['lead'], 'content': query['url'],
                           'candidate': candidates['idiom'][idx], 'definition': candidates['definition'][idx],
                           'score': "%.4f" % (distance), 'type': 'idiom'})
    final = pd.DataFrame(result)
    final.to_csv(f'{method}.csv', index=False)

if __name__ == '__main__':
    articles = pd.read_csv('articles_20.csv')
    candidates = pd.read_csv('idioms.csv')
    search('use', articles, candidates, top=1)