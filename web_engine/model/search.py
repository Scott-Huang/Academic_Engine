import spacy
import numpy as np
import warnings
from collections import defaultdict
from whoosh import index
from whoosh.fields import Schema, TEXT, STORED
from whoosh.qparser import QueryParser
from gensim.models import Word2Vec
from heapq import nlargest

warnings.filterwarnings('ignore')

INDEX_FOLDER = '../index_data'

nlp = spacy.load("en_core_web_md")
model = Word2Vec.load('../word2vec_model/word2vec.model').wv
schema = Schema(abstract=TEXT(),tags=STORED())
paper_index = index.open_dir(INDEX_FOLDER)
qp = QueryParser("abstract", schema=schema)
keyword_actual_freq = np.load('../keyword_freq.npy', allow_pickle='TRUE').item()
searcher = paper_index.searcher()

def compare_similarity(word1, word2):
    try:
        return abs(model.similarity(word1, word2))
    except:
        out = nlp(word1.replace('_', ' ')).similarity(nlp(word2.replace('_', ' ')))
        if not out:
            out = 0.05
        return out

def lemmatize(concept):
    doc = nlp(concept)
    txt = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return ('_'.join(txt)).replace('datum', 'data')

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm

def diversify(concepts, k, alpha):
    out = []
    concept_len = len(concepts)
    if concept_len == 0:
        return []

    sim_table = np.zeros((concept_len, k))
    concept_value = normalize(np.array(list(concepts.values())))
    concept_id = list(concepts.keys())
    
    for i in range(k):
        norm = np.linalg.norm(sim_table, ord=10, axis=1) ** 2 + 0.05
        norm = normalize(1/norm)
        temp = concept_value * (1-alpha) +  norm * alpha
        argmax = temp.argmax()

        concept = concept_id.pop(argmax)
        out.append(concept)
        sim_table = np.delete(sim_table,argmax,axis=0)
        concept_value = np.delete(concept_value,argmax,axis=0)

        for j in range(concept_len - i - 1):
            sim = compare_similarity(concept, concept_id[j])
            sim_table[j,i] = sim
    
    return out

def soft_freq(prob, freq):
    return prob / np.sqrt(freq)
def hard_freq(prob, freq):
    return prob / freq ** 1.05
freq_f = {'soft freq': soft_freq, 'hard freq': hard_freq}

def rank(word, freq_function='auto', k=9, diverse=True, alpha=0.5):
    '''Rank all concept words.'''
    
    # sum their probability
    r = defaultdict(int)
    word = word.strip()
    q = qp.parse(word)
    results = searcher.search(q, limit=15000)

    for hit in results:
        tags = eval(hit['tags'])
        score = np.log(hit.score)
        for keyword, key_position in tags.items():
            r[keyword] += score * len(key_position)

    if freq_function == 'auto':
        count = len(results)
        if count > 1000:
            freq_function = 'hard freq'
        elif count >= 10:
            freq_function = 'soft freq'
        else:
            freq_function = ''
    
    # further calculation
    r = dict(r)
    r.pop(word, '')

    topwords = defaultdict(int)
    for concept,probability in nlargest(800, r.items(), key=lambda x: x[1]):
        value = probability
        if freq_function in freq_f:
            value = freq_f[freq_function](value, keyword_actual_freq[concept])
        topwords[lemmatize(concept)] += value
    topwords = dict(topwords)
    topwords.pop('', '')
    
    if diverse and 1 > alpha and 0 < alpha:
        return diversify(topwords, k, alpha)

    out = [concept[0] for concept in nlargest(k, topwords.items(), key=lambda x: x[1])]
    return out

