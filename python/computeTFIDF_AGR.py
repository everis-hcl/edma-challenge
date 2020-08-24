from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from pathlib import Path
import pandas as pd
import json
import ipdb

postFilter = True

corpus_file = Path('./corpus/AGR/lemasAbstract/lemasAbstract.txt')
#corpus_file = Path('./corpus/AGR/EuropePMCAnnotations/EuropePMCAnnotations.txt')
base_docs = Path('./data/AGR/PMC_PMID_S2.csv')

base_tfidf = Path('./../data/TFIDFcorpus/AGR/lemasAbstract/')
#base_tfidf = Path('./TFIDFcorpus/AGR/EuropePMCAnnotations/')

S2_folder = Path('./data/AGR/Extended_AGR')

"""
##############################################################
# 1. Calculamos TFIDF sobre el dataset base únicamente
##############################################################
vocab_tfidf_file = base_tfidf.joinpath('BASE_vocab_tfidf.txt')
corpus_tfidf_file = base_tfidf.joinpath('BASE_corpus_tfidf.txt')
wdcorpus_tfidf_file = base_tfidf.joinpath('BASE_WDcorpus_tfidf.txt')

with corpus_file.open('r') as fin:
    corpus = fin.readlines()

#Prefiltering of documents not in BASE corpus
df = pd.read_csv(base_docs)
base_ids = set(df['S2ID'].values.tolist())
corpus = [el for el in corpus if el.strip().split()[0] in base_ids]
#S2ID to PMCID dictionary
map_dct = df[['S2ID','PMCID']].values.tolist()
map_dct = {el[0]:el[1] for el in map_dct}

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

##############################################################
# 3. Calculamos TFIDF para los autores (dataset extendido)
##############################################################
vocab_tfidf_file = base_tfidf.joinpath('AUTHOR_vocab_tfidf.txt')
corpus_tfidf_file = base_tfidf.joinpath('AUTHOR_corpus_tfidf.txt')
wdcorpus_tfidf_file = base_tfidf.joinpath('AUTHOR_WDcorpus_tfidf.txt')

with corpus_file.open('r') as fin:
    corpus = fin.readlines()

paper_ids = [el.strip().split()[0] for el in corpus]
paper_corpus = [el.strip().split()[2:] for el in corpus]

#Create corpus concatenating all papers for each author
author_files = sorted([d for d in S2_folder.iterdir() if d.name.startswith('Author') and d.name.endswith('.json')])

author_names = []
author_npapers = []
author_corpus = []

for af in author_files:
    with af.open() as fin:
        author_data = json.load(fin)
        author_names.append(author_data['name'])
        npapers = 0
        tokens = []
        for el in author_data['papers']:
            if el['paperId'] in paper_ids:
                npapers = npapers + 1
                tokens = tokens + paper_corpus[paper_ids.index(el['paperId'])]
        author_npapers.append(npapers)
        author_corpus.append(tokens)

dct = Dictionary(author_corpus)  # fit dictionary
corpus_bow = [dct.doc2bow(line) for line in author_corpus]  # convert corpus to BoW format

model = TfidfModel(corpus_bow)

corpus_tfidf = [model[el] for el in corpus_bow]
vocab_tfidf = [dct[el] for el in dct.keys()]

with vocab_tfidf_file.open('w') as fout:
    [fout.write(str(idx) + ':' + wd + '\n') for idx,wd in enumerate(vocab_tfidf)]

with corpus_tfidf_file.open('w') as fout:
    for an, anp, doc_tfidf in zip(author_names, author_npapers, corpus_tfidf):
        fout.write(an.replace(' ', '_') + ' ' + str(anp))
        for token in doc_tfidf:
            fout.write(' ' + str(token[0]) + ':' + str(token[1]))

        fout.write('\n')

with wdcorpus_tfidf_file.open('w') as fout:
    for an, anp, doc_tfidf in zip(author_names, author_npapers, corpus_tfidf):
        fout.write(an.replace(' ', '_') + ' ' + str(anp))
        for token in doc_tfidf:
            fout.write(' ' + vocab_tfidf[token[0]] + ':' + str(token[1]))

        fout.write('\n')


