from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from pathlib import Path
import pandas as pd
import ipdb

postFilter = True

#corpus_file = Path('./corpus/AGR/lemasAbstract/lemasAbstract.txt')
corpus_file = Path('./corpus/BIO/lemasWithProcedure/lemasWithProcedure.txt')

#base_tfidf = Path('./TFIDFcorpus/AGR/lemasAbstract/')
base_tfidf = Path('./TFIDFcorpus/BIO/lemasWithProcedure/')

##############################################################
# 1. Calculamos TFIDF sobre el dataset base únicamente
##############################################################
vocab_tfidf_file = base_tfidf.joinpath('BASE_vocab_tfidf.txt')
corpus_tfidf_file = base_tfidf.joinpath('BASE_corpus_tfidf.txt')
wdcorpus_tfidf_file = base_tfidf.joinpath('BASE_WDcorpus_tfidf.txt')

with corpus_file.open('r') as fin:
    corpus = fin.readlines()

#Prefiltering of documents not in BASE corpus
corpus = [el for el in corpus if el.strip().split()[0].startswith('Bio-protocol')]

ids = [el.strip().split()[0] for el in corpus]
corpus = [el.strip().split()[2:] for el in corpus]

dct = Dictionary(corpus)  # fit dictionary
corpus_bow = [dct.doc2bow(line) for line in corpus]  # convert corpus to BoW format

model = TfidfModel(corpus_bow)
corpus_tfidf = [model[el] for el in corpus_bow] 
vocab_tfidf = [dct[el] for el in dct.keys()]

with vocab_tfidf_file.open('w') as fout:
    [fout.write(str(idx) + ':' + wd + '\n') for idx,wd in enumerate(vocab_tfidf)]

with corpus_tfidf_file.open('w') as fout:
    for doc_id, doc_tfidf in zip(ids, corpus_tfidf):
        fout.write(doc_id)
        for token in doc_tfidf:
            fout.write(' ' + str(token[0]) + ':' + str(token[1]))

        fout.write('\n')

with wdcorpus_tfidf_file.open('w') as fout:
    for doc_id, doc_tfidf in zip(ids, corpus_tfidf):
        fout.write(doc_id)
        for token in doc_tfidf:
            fout.write(' ' + vocab_tfidf[token[0]] + ':' + str(token[1]))

        fout.write('\n')

"""
##############################################################
# 2. Calculamos TFIDF sobre el dataset extendido únicamente
##############################################################
vocab_tfidf_file = base_tfidf.joinpath('EXT_vocab_tfidf.txt')
corpus_tfidf_file = base_tfidf.joinpath('EXT_corpus_tfidf.txt')
wdcorpus_tfidf_file = base_tfidf.joinpath('EXT_WDcorpus_tfidf.txt')

with corpus_file.open('r') as fin:
    corpus = fin.readlines()

ids = [el.strip().split()[0] for el in corpus]
corpus = [el.strip().split()[2:] for el in corpus]

dct = Dictionary(corpus)  # fit dictionary
corpus_bow = [dct.doc2bow(line) for line in corpus]  # convert corpus to BoW format

model = TfidfModel(corpus_bow)

with corpus_file.open('r') as fin:
    corpus = fin.readlines()
corpus = [el for el in corpus if el.strip().split()[0] in base_ids]
ids = [el.strip().split()[0] for el in corpus]
corpus = [el.strip().split()[2:] for el in corpus]
corpus_bow = [dct.doc2bow(line) for line in corpus]

corpus_tfidf = [model[el] for el in corpus_bow]
vocab_tfidf = [dct[el] for el in dct.keys()]

with vocab_tfidf_file.open('w') as fout:
    [fout.write(str(idx) + ':' + wd + '\n') for idx,wd in enumerate(vocab_tfidf)]

with corpus_tfidf_file.open('w') as fout:
    for doc_id, doc_tfidf in zip(ids, corpus_tfidf):
        fout.write(doc_id + ' ' + map_dct[doc_id])
        for token in doc_tfidf:
            fout.write(' ' + str(token[0]) + ':' + str(token[1]))

        fout.write('\n')

with wdcorpus_tfidf_file.open('w') as fout:
    for doc_id, doc_tfidf in zip(ids, corpus_tfidf):
        fout.write(doc_id + ' ' + map_dct[doc_id])
        for token in doc_tfidf:
            fout.write(' ' + vocab_tfidf[token[0]] + ':' + str(token[1]))

        fout.write('\n')

"""
