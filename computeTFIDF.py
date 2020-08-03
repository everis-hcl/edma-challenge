from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from pathlib import Path
import pandas as pd
import ipdb

extended = False

corpus_file = Path('./corpus/AGR/lemasAbstract/lemasAbstract.txt')
#corpus_file = Path('./corpus/AGR/EuropePMCAnnotations/EuropePMCAnnotations.txt')
base_docs = Path('./data/AGR/PMC_PMID_S2.csv')

vocab_tfidf_file = Path('./vocab_tfidf.txt')
corpus_tfidf_file = Path('./corpus_tfidf.txt')

with corpus_file.open('r') as fin:
    corpus = fin.readlines()

if not extended:
    df = pd.read_csv(base_docs)
    base_ids = set(df['S2ID'].values.tolist())
    corpus = [el for el in corpus if el.strip().split()[0] in base_ids]

ids = [el.strip().split()[0] for el in corpus]
corpus = [el.strip().split()[2:] for el in corpus]

dct = Dictionary(corpus)  # fit dictionary
corpus_bow = [dct.doc2bow(line) for line in corpus]  # convert corpus to BoW format

model = TfidfModel(corpus_bow)
corpus_tfidf = [model[el] for el in corpus_bow] 
vocab_tfidf = [dct[el] for el in dct.keys()]

with vocab_tfidf_file.open('w') as fout:
    [fout.write(wd + '\n') for wd in vocab_tfidf]

with corpus_tfidf_file.open('w') as fout:
    for doc_id, doc_tfidf in zip(ids, corpus_tfidf):
        fout.write(doc_id)
        for token in doc_tfidf:
            fout.write(' ' + str(token[0]) + ':' + str(token[1]))

        fout.write('\n')



