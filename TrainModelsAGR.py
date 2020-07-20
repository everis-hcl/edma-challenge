import pandas as pd
import json
import re
import regex as javare
import configparser
import time
from pathlib import Path
from gensim import corpora
from gensim.utils import check_output
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
from scipy import sparse
import scipy
import gensim
from gensim.models.coherencemodel import CoherenceModel
import ipdb

from utils import printgr, printred, printmag
from dbManager.base_dm_sql import BaseDMsql

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

def file_len(fname):
    with fname.open('r',encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1

"""
============================================================
Paths to relevant folders and files
============================================================
"""
Path2data = Path('./data/AGR')
PMClist_file = Path2data.joinpath('PMC_list.txt')
csv_file = Path2data.joinpath('PMC_PMID_S2.csv')
annotations_folder = Path2data.joinpath('EuropePMC_annotations')
S2_folder = Path2data.joinpath('Extended_AGR')
csv_file_extended = Path2data.joinpath('PMID_S2_extendedAGR.csv')

Path2corpus = Path('./corpus/AGR')
lemas_file = Path2corpus.joinpath('AGR_lemas.csv')
Path2models = Path('./models/AGR')

"""
============================================================
Variables to adjust what part of the code will be executed
============================================================
"""

lemmatization = False
generateCorpus = False
trainMany = True
coherence = False

cf = configparser.ConfigParser()
cf.read('config.cf')

"""
============================================================
Lemmatization of the AGR data
We read the relevant files from the CSV file and generate
a lemmatized dataset to share with Everis
============================================================
"""
if lemmatization:

    #Conectamos a la Base de Datos de Semantic Scholar
    dbCONNECTOR = cf.get('DB', 'dbCONNECTOR')
    dbNAME = cf.get('DB', 'dbNAME')
    dbUSER = cf.get('DB', 'dbUSER')
    dbPASS = cf.get('DB', 'dbPASS')
    dbSERVER = cf.get('DB', 'dbSERVER')
    dbSOCKET = cf.get('DB', 'dbSOCKET')
    DM = BaseDMsql(db_name=dbNAME, db_connector=dbCONNECTOR, path2db=None,
               db_server=dbSERVER, db_user=dbUSER, db_password=dbPASS,
               unix_socket=dbSOCKET)

    printgr('Reading Agriculture data from database')
    AGR_df = pd.read_csv(csv_file_extended, low_memory=False, dtype=str)
    AGR_S2 = AGR_df['S2paperID'].values.tolist()
    AGR_df = pd.DataFrame()
    for S2id in AGR_S2:
        dfaux = DM.readDBtable('S2papers', limit=None, selectOptions='S2paperID, title, paperAbstract',
                                filterOptions='S2paperID="'+S2id+'"')
        AGR_df = AGR_df.append(dfaux, ignore_index = True)
    print('Agriculture data loaded, #papers:', len(AGR_df))

    from lemmatizer.ENlemmatizer import ENLemmatizer
    lemmas_server = cf.get('Lemmatizer', 'server')
    stw_file = Path(cf.get('Lemmatizer', 'default_stw_file'))
    dict_eq_file = Path(cf.get('Lemmatizer', 'default_dict_eq_file'))
    POS = cf.get('Lemmatizer', 'POS')
    concurrent_posts = int(cf.get('Lemmatizer', 'concurrent_posts'))
    removenumbers = cf.get('Lemmatizer', 'removenumbers') == 'True'
    keepSentence = cf.get('Lemmatizer', 'keepSentence') == 'True'
    init_time = time.time()
    #Initialize lemmatizer
    ENLM = ENLemmatizer(lemmas_server=lemmas_server, stw_file=stw_file,
                    dict_eq_file=dict_eq_file, POS=POS, removenumbers=removenumbers,
                    keepSentence=keepSentence)

    AGR_df['alltext'] = AGR_df['title'] + '. ' + AGR_df['paperAbstract']
    AGR_df['alltext'] = AGR_df['alltext'].apply(clean_utf8)
    lemasBatch = ENLM.lemmatizeBatch(AGR_df[['S2paperID', 'alltext']].values.tolist(),
                                                processes=concurrent_posts)
    #Remove entries that where not lemmatized correctly
    lemasBatch = [[el[0], clean_utf8(el[1])] for el in lemasBatch]
    print('Successful lemmatized documents: {len(lemasBatch)}')
    elapsed_time = time.time() - init_time
    print(f'Elapsed Time (seconds): {time.strftime("%H:%M:%S", time.gmtime(elapsed_time))}')
    AGR_df['LEMAS'] = [el[1] for el in lemasBatch]

    #Save lemas to file
    if not Path2corpus.exists():
        Path2corpus.mkdir(parents=True)
    AGR_df = AGR_df.replace(np.nan, '', regex=True)
    AGR_df.to_csv(lemas_file, index=False)


"""
============================================================
Generate Corpus for Mallet
============================================================
"""
if generateCorpus:
    min_lemas = int(cf.get('CorpusGeneration', 'min_lemas'))
    no_below=int(cf.get('CorpusGeneration','no_below'))
    no_above=float(cf.get('CorpusGeneration','no_above'))
    keep_n=int(cf.get('CorpusGeneration','keep_n'))
    token_regexp=javare.compile(cf.get('CorpusGeneration','token_regexp'))
    mallet_path = Path(cf.get('TM','mallet_path'))

    #Initialize object for applying equivalences and stopwords
    stw_file = Path(cf.get('Lemmatizer', 'default_stw_file'))
    corpus_eqs = Path(cf.get('AGR', 'corpus_eq_file'))
    corpus_stw = Path(cf.get('AGR', 'corpus_stw_file'))

    from lemmatizer.ENlemmatizer import stwEQcleaner
    stwEQ = stwEQcleaner(stw_files=[stw_file, corpus_stw], dict_eq_file=corpus_eqs)
    
    #############################################################
    # Generate corpus with lemas for the abstracts
    #############################################################
    AGR_df = pd.read_csv(lemas_file, low_memory=False, dtype=str)
    id_lema = AGR_df[['S2paperID', 'LEMAS']].values.tolist()

    #corpus_name = input('Introduzca un nombre para el corpus a generar: ')
    #corpus_dir = Path('./datos/corpus_'+corpus_name)
    corpus_dir = Path2corpus.joinpath('lemasAbstract')
    print('El corpus ampliado con los Abstracts lematizados se guardará en el directorio', corpus_dir)
    corpus_dir.mkdir()

    #To make sure all papers in base dataset are kept
    df = pd.read_csv(csv_file)
    S2_base = set(df['S2ID'].values.tolist())

    import_config = corpus_dir.joinpath('import.config')
    with import_config.open('w', encoding='utf8') as fout:
        fout.write('min_lemas = ' + str(min_lemas) + '\n')
        fout.write('no_below = ' + str(no_below) + '\n')
        fout.write('no_above = ' + str(no_above) + '\n')
        fout.write('keep_n = ' + str(keep_n) + '\n')
        fout.write('token_regexp = ' + str(token_regexp) + '\n')

    dictionary = corpora.Dictionary()
    id_lema = [[el[0], ' '.join(token_regexp.findall(str(el[1]).replace('\n',' ').strip()))]
                        for el in id_lema]
    id_lema = [[el[0], stwEQ.cleanstr(el[1]).split()] for el in id_lema]
    all_lemas = [el[1] for el in id_lema if len(el[1])>=min_lemas or el[0] in S2_base]
    dictionary.add_documents(all_lemas)

    #Remove words that appear in less than no_below documents, or in more than
    #no_above, and keep at most keep_n most frequent terms, keep track of removed
    #words for debugging purposes
    all_words = [dictionary[idx] for idx in range(len(dictionary))]
    dictionary.filter_extremes(no_below=no_below, no_above=no_above, keep_n=keep_n)
    kept_words = set([dictionary[idx] for idx in range(len(dictionary))])

    corpus_file = corpus_dir.joinpath('lemasAbstract.txt')
    corpus_mallet = corpus_dir.joinpath('lemasAbstract.mallet')
    id_lema = [[el[0], [tk for tk in el[1] if tk in kept_words]] for el in id_lema]
    id_lema = [[el[0], stwEQ.cleanstr(' '.join(el[1])).split()] for el in id_lema]
    id_lema = [el for el in id_lema if len(el[1])>=min_lemas or el[0] in S2_base]
    #Check if all papers in base dataset are in the extended corpus
    #since they could have been removed if the lemmatizer did not provide a valid
    #output for them
    df = pd.read_csv(csv_file)
    S2_base = set(df['S2ID'].values.tolist())
    S2_in = [el for el in id_lema if el[0] in S2_base]
    if len(S2_in)==len(S2_base):
        printgr('Todos los papers del dataset base se han incorporado al dataset extendido de Abstracts lematizados')
    else:
        printred('Se han perdido papers del dataset base ' + str(len(S2_in)) + ' / ' + str(len(S2_base)))

    print('Generating corpus, #papers:', len(id_lema))

    with corpus_file.open('w', encoding='utf-8') as fout:
        [fout.write(el[0] + ' 0 ' + ' '.join(el[1]) + '\n') for el in id_lema]

    token_regexp=cf.get('CorpusGeneration','token_regexp')
    cmd = str(mallet_path) + \
              ' import-file --preserve-case --keep-sequence ' + \
              '--remove-stopwords --token-regex "' + token_regexp + '" ' + \
              '--input %s --output %s'
    cmd = cmd % (corpus_file, corpus_mallet)

    try:
        print(f'-- -- Running command {cmd}')
        check_output(args=cmd, shell=True)
    except:
        print('-- -- Mallet failed to import data. Revise command')

"""
============================================================
Generate Topic models
============================================================
"""
if trainMany:
    mallet_path = Path(cf.get('TM', 'mallet_path'))
    ntopicsArray = [int(el) for el in cf.get('TM', 'num_topics_many').split(',')]
    alphasArray = [float(el) for el in cf.get('TM', 'alphas_many').split(',')]
    intervalArray = [int(el) for el in cf.get('TM', 'optimize_interval_many').split(',')]
    num_threads = int(cf.get('TM', 'num_threads'))
    num_iterations = int(cf.get('TM', 'num_iterations'))
    doc_topic_thr = float(cf.get('TM', 'doc_topic_thr'))

    available_corpus = sorted([d for d in Path2corpus.iterdir() if d.is_dir()])
    display_corpus = [str(idx) +'. ' + d.name for idx,d in enumerate(available_corpus)]
    [print(option) for option in display_corpus]
    selection = input('Select corpus for topic modeling: ')
    path_corpus = available_corpus[int(selection)]

    models_dir = Path2models.joinpath(path_corpus.name)
    #models_dir.mkdir(parents=True)

    #Iterate model training
    for ntopics in ntopicsArray:
        for alpha in alphasArray:
            for interval in intervalArray:
                path_model =  models_dir.joinpath('Ntpc_'+str(ntopics)+'_a_'+str(alpha)+\
                                     '_interval_'+str(interval))
                path_model.mkdir()

                config_file = path_model.joinpath('train.config')
                with config_file.open('w', encoding='utf8') as fout:
                    fout.write('input = ' + path_corpus.joinpath(path_corpus.name+'.mallet').as_posix() + '\n')
                    fout.write('num-topics = ' + str(ntopics) + '\n')
                    fout.write('alpha = ' + str(alpha) + '\n')
                    fout.write('optimize-interval = ' + str(interval) + '\n')
                    fout.write('num-threads = ' + str(num_threads) + '\n')
                    fout.write('num-iterations = ' + str(num_iterations) + '\n')
                    fout.write('output-doc-topics = ' + \
                        path_model.joinpath('doc-topics.txt').as_posix() + '\n')
                    fout.write('word-topic-counts-file = ' + \
                        path_model.joinpath('word-topic-counts.txt').as_posix() + '\n')
                    fout.write('output-topic-keys = ' + \
                        path_model.joinpath('topickeys.txt').as_posix() + '\n')

                cmd = str(mallet_path) + ' train-topics --config ' + str(config_file)

                try:
                    print(f'-- -- Training mallet topic model. Command is {cmd}')
                    check_output(args=cmd, shell=True)
                except:
                    print('-- -- Model training failed. Revise command')
                
                #Sparsify thetas and build vocabulary files
                sparse_thr=3e-3
                thetas_file = path_model.joinpath('doc-topics.txt')
                cols = [k for k in np.arange(2,ntopics+2)]
                thetas32 = np.loadtxt(thetas_file, delimiter='\t', dtype=np.float32, usecols=cols)
                allvalues = np.sort(thetas32.flatten())
                step = int(np.round(len(allvalues)/1000))
                plt.semilogx(allvalues[::step], (100/len(allvalues))*np.arange(0,len(allvalues))[::step])
                plt.semilogx([sparse_thr, sparse_thr], [0,100], 'r')
                plot_file = path_model.joinpath('thetas_dist.pdf')
                plt.savefig(plot_file)
                plt.close()

                #sparsify thetas
                thetas32[thetas32<sparse_thr] = 0
                thetas32 = normalize(thetas32,axis=1,norm='l1')
                thetas32_sparse = sparse.csr_matrix(thetas32, copy=True)
                scipy.sparse.save_npz(path_model.joinpath('thetas_sparse.npz'), thetas32_sparse)

                #Create vocabulary files
                wtcFile = path_model.joinpath('word-topic-counts.txt')
                vocab_size = file_len(wtcFile)
                vocab = []
                term_freq = np.zeros((vocab_size,))

                with wtcFile.open('r', encoding='utf8') as fin:
                    for i,line in enumerate(fin):
                        elements = line.split()
                        vocab.append(elements[1])
                        for counts in elements[2:]:
                            tpc = int(counts.split(':')[0])
                            cnt = int(counts.split(':')[1])
                            term_freq[i] += cnt

                with path_model.joinpath('vocab.txt').open('w', encoding='utf8') as fout:
                    [fout.write(el+'\n') for el in vocab]
                with path_model.joinpath('vocab_freq.txt').open('w', encoding='utf8') as fout:
                    [fout.write(el[0]+'\t'+str(int(el[1]))+'\n') for el in zip(vocab,term_freq)]

                #Obtain topic-based representation for papers and authors
                with thetas_file.open() as fin:
                    S2_ids = [el.split('\t')[1] for el in fin.readlines()]
                S2_ids = {el:idx for idx,el in enumerate(S2_ids)}
                
                #Paper representation
                papers_df = pd.read_csv(csv_file)
                
                with open(path_model.joinpath('paper_topics.txt'), 'w') as fout:
                    for el in papers_df.values.tolist():
                        display_tpcs = str(np.array(thetas32_sparse[S2_ids[el[2]],].todense())[0].tolist())[1:-1]
                        fout.write(el[0]+'\t'+display_tpcs+'\n')

                #Author representation
                author_files = sorted([d for d in S2_folder.iterdir() if d.name.startswith('Author') and d.name.endswith('.json')])
                with open(path_model.joinpath('author_topics.txt'), 'w') as fout:
                    for af in author_files:
                        with af.open() as fin:
                            author_data = json.load(fin)
                        paper_pos = [S2_ids[el['paperId']] for el in author_data['papers'] if el['paperId'] in S2_ids]
                        n_papers = len(paper_pos)
                        topic_submatr = thetas32_sparse[paper_pos,]
                        author_topics = np.array(np.mean(topic_submatr, axis=0))[0]
                        display_tpcs = str(author_topics.tolist())[1:-1]
                        fout.write(author_data['name']+'\t'+str(n_papers)+'\t'+display_tpcs+'\n')

"""
============================================================
Calculate coherence of lemasAbstract model
============================================================
"""

if coherence:
    available_models = sorted([d for d in Path2models.iterdir() if d.is_dir()])
    display_models = [str(idx) +'. ' + d.name for idx,d in enumerate(available_models)]
    [print(option) for option in display_models]
    selection = input('Select model for coherence calculation: ')
    path_model = available_models[int(selection)]

    all_models = [d for d in path_model.iterdir() if d.is_dir()]
    all_models = sorted(all_models, key=lambda x: int(x.name.split('_')[1]))

    #Load corpus
    path_corpus = Path2corpus.joinpath(path_model.name).joinpath(path_model.name+'.txt')
    with path_corpus.open('r', encoding='utf8') as fin:
        corpus = [el.strip().split()[2:] for el in fin.readlines()]
    D = gensim.corpora.Dictionary(corpus)
    kept_words = set([D[idx] for idx in range(len(D))])

    tpcs = []
    coherence_cv = []
    for model in all_models:
        print(model)
        topickeys_file = model.joinpath('topickeys.txt')
        ntpc = int(model.name.split('_')[1])
        tpcs.append(ntpc)
        # Cargar los términos que definen cada tópico
        topic_keys = np.loadtxt(topickeys_file, delimiter='\t', dtype=str, usecols=2)
        topics = [(t.split()) for t in topic_keys]
        #Remove topic words not in dictionary
        for idx, tpc_words in enumerate(topics):
            topics[idx] = [el for el in tpc_words if el in kept_words]
        #cm = CoherenceModel(topics=topics, corpus=corpus_bow, dictionary=D, coherence='u_mass')
        cm = CoherenceModel(topics=topics, texts=corpus, dictionary=D, coherence='c_v', processes=15)
        coherence = cm.get_coherence()
        coherence_cv.append(coherence)
    print(tpcs)
    print(coherence_cv)


