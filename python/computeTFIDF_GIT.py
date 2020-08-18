from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from pathlib import Path
import re
import regex as javare
import configparser
import json
import ipdb
import langid
from lemmatizer.ENlemmatizer import ENLemmatizer

corpus_folder = Path('./data/GIT/GitFiles')

base_tfidf = Path('./TFIDFcorpus/GIT/lemasReadme/')

try:
    # UCS-4
    regex = re.compile('[\U00010000-\U0010ffff]')
except re.error:
    # UCS-2
    regex = re.compile('[\uD800-\uDBFF][\uDC00-\uDFFF]')

def clean_utf8(rawdata):
    try:
        cleaned_data = regex.sub(' ', rawdata)
    except:
        cleaned_data = ''
    return cleaned_data

##############################################################
# 1. Carga de datos
##############################################################

repos = [f for f in corpus_folder.iterdir() if f.name.endswith('json')]

all_repos = []
for repo in repos:
    with repo.open() as fin:
        repo_data = json.load(fin)
    if len(repo_data['readMe'])>100 and langid.classify(repo_data['readMe'])[0]=='en':
        all_repos.append([repo.name, repo_data['owner'], repo_data['readMe']])

##############################################################
# 2. Lematización
##############################################################

cf = configparser.ConfigParser()
cf.read('config.cf')

lemmas_server = cf.get('Lemmatizer', 'server')
stw_file = Path(cf.get('Lemmatizer', 'default_stw_file'))
dict_eq_file = Path(cf.get('Lemmatizer', 'default_dict_eq_file'))
POS = cf.get('Lemmatizer', 'POS')
concurrent_posts = int(cf.get('Lemmatizer', 'concurrent_posts'))
removenumbers = cf.get('Lemmatizer', 'removenumbers') == 'True'
keepSentence = cf.get('Lemmatizer', 'keepSentence') == 'True'

#Initialize lemmatizer
ENLM = ENLemmatizer(lemmas_server=lemmas_server, stw_file=stw_file,
                    dict_eq_file=dict_eq_file, POS=POS, removenumbers=removenumbers,
                    keepSentence=keepSentence)

all_repos = [[el[0], el[1], clean_utf8(el[2])] for el in all_repos]
lemasBatch = ENLM.lemmatizeBatch([[el[0], el[2]] for el in all_repos], processes=1)
lemasBatch = [[el[0], clean_utf8(el[1])] for el in lemasBatch]

all_repos = [[el0[0], el0[1], el1[1]] for el0, el1 in zip(all_repos, lemasBatch) if len(el1[1])]

##############################################################
# 3. Calculamos TFIDF para los repositorios
##############################################################
vocab_tfidf_file = base_tfidf.joinpath('BASE_vocab_tfidf.txt')
corpus_tfidf_file = base_tfidf.joinpath('BASE_corpus_tfidf.txt')
wdcorpus_tfidf_file = base_tfidf.joinpath('BASE_WDcorpus_tfidf.txt')

ids = [el[0] for el in all_repos]
corpus = [el[2].strip().split() for el in all_repos]

dct = Dictionary(corpus)  # fit dictionary
dct.filter_extremes(no_below=1, no_above=0.6)
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

##############################################################
# 3. Calculamos TFIDF para los autores
##############################################################
vocab_tfidf_file = base_tfidf.joinpath('AUTHOR_vocab_tfidf.txt')
corpus_tfidf_file = base_tfidf.joinpath('AUTHOR_corpus_tfidf.txt')
wdcorpus_tfidf_file = base_tfidf.joinpath('AUTHOR_WDcorpus_tfidf.txt')

authors = sorted(list(set([el[1] for el in all_repos])))

ids = []
n_repos = []
corpus = []

for author in authors:
    ids.append(author)
    text = ''
    nrep = 0
    for repo in all_repos:
        if repo[1]==author:
            text = text + repo[2]
            nrep = nrep + 1
    corpus.append(text)
    n_repos.append(nrep)

corpus = [el.strip().split() for el in corpus]

dct = Dictionary(corpus)  # fit dictionary
dct.filter_extremes(no_below=1, no_above=0.6)
corpus_bow = [dct.doc2bow(line) for line in corpus]  # convert corpus to BoW format

model = TfidfModel(corpus_bow)
corpus_tfidf = [model[el] for el in corpus_bow]
vocab_tfidf = [dct[el] for el in dct.keys()]

with vocab_tfidf_file.open('w') as fout:
    [fout.write(str(idx) + ':' + wd + '\n') for idx,wd in enumerate(vocab_tfidf)]

with corpus_tfidf_file.open('w') as fout:
    for doc_id, nrep, doc_tfidf in zip(ids, n_repos, corpus_tfidf):
        fout.write(doc_id + ' ' + str(nrep))
        for token in doc_tfidf:
            fout.write(' ' + str(token[0]) + ':' + str(token[1]))

        fout.write('\n')

with wdcorpus_tfidf_file.open('w') as fout:
    for doc_id, nrep, doc_tfidf in zip(ids, n_repos, corpus_tfidf):
        fout.write(doc_id + ' ' + str(nrep))
        for token in doc_tfidf:
            fout.write(' ' + vocab_tfidf[token[0]] + ':' + str(token[1]))

        fout.write('\n')



