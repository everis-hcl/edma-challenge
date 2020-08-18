"""
map2tax implements a labeler of documents based on list of tokens.

It assumes that three input files are available:

- A document describing topics as list of tokens
- A docuemnt describing documents as weighted combinations of topics
- A document describind authors as weighted combinationsof of topics.

The code returns three files, providing labels for topics, documents and
authors

The core of the automatic labelling process is an adaptation of code taken from
the NETL library,
(https://github.com/sb1992/NETL-Automatic-Topic-Labelling-)
with apache license 2.0.

Most code is a modification of /model_run/supervised_labels.py. Some methods
have been significantly changed to speedup.

USAGE: (see bou2tax --help for details)

bow2tax.py [-h] [--source SOURCE] [--tax TAX] [--path2tax PATH2TAX]
                  [--model MODEL] [--output OUTPUT] [--tmax TMAX]

optional arguments:
  -h, --help           show this help message and exit
  --source SOURCE      path to the source data folder
  --tax TAX            taxonomy (esv or agr)
  --path2tax PATH2TAX  path to the xlsx file containing the taxonomy (relative
                       to source)
  --model MODEL        path to the model to process (relative to source)
  --output OUTPUT      path to the output folder
  --tmax TMAX          maximum number of terms to take from each topic
"""

import pathlib
import numpy as np
from sklearn.preprocessing import normalize
import heapq
import csv
import argparse
from time import time

# Local imports
import my_supervised_labels


def file_len(fname):
    """
    Returns the number of lines in a text file.
    """
    with fname.open('r', encoding='utf8') as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def read_topics(path):
    """
    Reads topic information from the topic_counts file returned by the LDA
    topic extractor.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the topic_counts file

    Returns
    -------
    vocab : list
        List of terms
    term_freq : numpy.ndarray
        Array with the frequency (no. of appearances) of each term in vocab
    betas : numpy.ndarray
        Normalized topic matrix
    """

    # Number of topics.
    # This number is read from the folder name. This should be replaced by a
    # more robust reading from the file content.
    n_topics = int(path.parent.stem.split('_')[1])

    # Path to the file with the topic model
    vocab_size = file_len(path)
    betas = np.zeros((n_topics, vocab_size))
    vocab = []
    term_freq = np.zeros((vocab_size,))

    with path.open('r', encoding='utf8') as fin:
        for i, line in enumerate(fin):
            elements = line.split()
            vocab.append(elements[1])
            for counts in elements[2:]:
                tpc = int(counts.split(':')[0])
                cnt = int(counts.split(':')[1])
                betas[tpc, i] += cnt
                term_freq[i] += cnt
    betas = normalize(betas, axis=1, norm='l1')

    return vocab, term_freq, betas


def read_matrix_from_txt(path):
    """
    Reads matrix data from a text file.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to the text file

    Returns
    -------
    Y : numpy.ndarray
        Data matrix
    """

    # Number of topics.
    # This number is read from the folder name. This should be replaced by a
    # more robust reading from the file content.
    n_cols = int(path.parent.stem.split('_')[1])

    # Path to the file with the topic model
    n_items = file_len(path)
    Y = np.zeros((n_items, n_cols))
    ids = []

    with path.open('r', encoding='utf8') as fin:
        for i, line in enumerate(fin):
            elements = line.split()
            n_comps = len(elements)

            # Number of components that correspond to the id.
            # This must be computed because in some txt files the id can
            # consist of a variable number of space-separated terms
            # (e.g. a name and one or more surnames)
            n_name = n_comps - n_cols
            # Add id components
            ids.append(' '.join(elements[:n_name]))
            # Add matrix row
            for j, counts in enumerate(elements[n_name:]):
                # Note that I ignore the last character of counts, which is ','
                Y[i, j] += float(counts[:-1])

    Y = normalize(Y, axis=1, norm='l1')

    return ids, Y


def top_tokens_4_NETL(X, vocab, tmax):
    """
    Given a (N, M) matrix and a vocabulary list of N terms, returns a list of
    the terms corresponding to the top k_max values of each matrix row, ready
    to be used by NETL library

    For instance, if the top-3 values in the first row of X lie in columns
    8, 3 and 5, the firts rows of the output list contain the list
    [vocab[8], vocab[3], vocab[5]]

    Parameters
    ----------
    X : numpy.ndarray
        A bidimensional array
    vocab : list
        The vocabulary. The number of elements must be equal to the number of
        rows in X
    tmax : int
        Number of terms to be taken from each row

    Returns
    -------
    Y : list of list
        The first row is a header required by the NETL library
        The first element in each list is an integer id, also requiered by NETL
    """

    # #############################
    # Get top terms from each topic

    # Top terms from each topic.
    n_rows, n_tokens = X.shape

    # The first element in the list is the NETL header
    Y = [["topic_id"] + [f"term{k}" for k in range(tmax)]]
    for i in range(n_rows):
        t_weights = zip(vocab, X[i])
        topk = heapq.nlargest(tmax, t_weights, lambda z: z[1])
        topk = [i] + [el[0] for el in topk]
        Y.append(topk)

    return Y


def save_data(path2out, data, label):
    """
    Saves data into files to be used by the NETL library
    """

    # Write topics to csv file
    with open(path2out / f'tok_{label}.csv', "w", newline="",
              encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(data)

    return


# #####
# Start

t0 = time()

# ####################
# Command-line options

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=str, default='../source_data',
                    help='path to the source data folder')
parser.add_argument('--tax', type=str, help="taxonomy (esv or agr)",
                    default='esv')
parser.add_argument(
    '--path2tax', type=str, default=None,
    help='path to the xlsx file containing the taxonomy (relative to source)')
parser.add_argument('--model', type=str, default=None,
                    help="path to the model to process (relative to source)")
parser.add_argument('--output', type=str, help="path to the output folder")
parser.add_argument('--tmax', type=int, default=20,
                    help="maximum number of terms to take from each topic")
args = parser.parse_args()

if args.output is None or args.model is None:
    print("-- ERROR: Missing arguments. Type python map2tax --h for help")
    exit()

# ###############
# Input arguments

# Taxonomy
taxonomy = args.tax

# Maximum number of terms to take from each topic
tmax = args.tmax

# Path to the source data folder
path2source = pathlib.Path(args.source)
# Path to the folder containing the topic model files
path2model = path2source / args.model
# Path to folder that will store resuts from all executions
path2out = pathlib.Path(args.output) / f"{path2model.name}_tmax_{tmax}"
# Path to the file containing the taxonomy
if args.path2tax is not None:
    path2tax = path2source / args.path2tax
elif taxonomy == 'esv':
    path2tax = path2source / 'taxonomy' / 'EuroSciVoc_Category-All-Report.xlsx'
else:
    path2tax = path2source / 'taxonomy' / 'agrovocab.yml'

# Path to the file containing all wikipedia titles
# path2wikifile = '../source_data/enwiki-20200801-redirect.sql'
# (taken from https://dumps.wikimedia.org/enwiki/20200801
#                    /enwiki-20200801-all-titles-in-ns0.gz)
path2wikifile = (path2source / 'wikisources'
                 / 'enwiki-20200801-all-titles-in-ns0')

# #########
# Set paths

# Set relevant paths
path2topics = path2model / 'word-topic-counts.txt'
path2authors = path2model / 'author_topics.txt'
path2papers = path2model / 'paper_topics.txt'

path2out.mkdir(parents=True, exist_ok=True)

# #########
# Read data

print(" ********************")
print(" *** NETL labeler ***\n")

print('Loading source data...')

# Read topic model
print('-- Loading topics...')
vocab, term_freq, betas = read_topics(path2topics)
# Read matrix of docs
print('-- Loading doc-topic representations ...')
paper_ids, doc_topic = read_matrix_from_txt(path2papers)
# Read matrix of authors
print('-- Loading author-topic representations ...')
author_ids, author_topic = read_matrix_from_txt(path2authors)

# ###############
# Transforma data
author_token = author_topic @ betas
doc_token = doc_topic @ betas

# #############################
# Get top terms from each topic

# Top terms from each topic, doc or author
print(f'-- Computing top-{tmax} terms...')
topics = top_tokens_4_NETL(betas, vocab, tmax)
docs = top_tokens_4_NETL(doc_token, vocab, tmax)
authors = top_tokens_4_NETL(author_token, vocab, tmax)
print('-- Saving all results into files')
save_data(path2out, topics, 'topics')
save_data(path2out, docs, 'docs')
save_data(path2out, authors, 'authors')
print(f'Top terms for topics, docs and authors saved in {path2out}')
print(f'These terms will be the inputs to the NETL labeler')

# #########
# NETL Zone

# This section is an adaptation of get_labels.py, taken from the NETL library,
# written by Shraey Bhatia, and update by Sihwa Park for python 3.
#
# It is the script to generate candidate labels, unsupervised best labels and
# labels from SVM ranker supervised model.
# Update the parameters in this script.
# Also after you download the files mentioned in readme and if you keep them in
# different path change it over here.

path2ntl = path2source / "netl_support_files"

# This is precomputed pagerank model needed to genrate pagerank features.
pagerank_model = path2ntl / "pagerank-titles-sorted.txt"
# SVM rank classify. After you download SVM Ranker classify gibve the path of
# svm_rank_classify here
svm_classify = path2ntl / "svm_rank_classify"
# This is trained supervised model on the whole our dataset. Run train
# train_svm_model.py if you want a new model on different dataset.
pretrained_svm_model = path2ntl / "svm_model"

# Number of supervised labels needed. Should be less than the candidate labels.
num_sup_labels = 3

# Input data folder
data_folder = path2out     # f'output_20/{model}'

# We will call the NETL labeler for each one of these datasets:
tags = ['topics', 'docs', 'authors']
taxonomy_name = {'esv': 'EuroSciVoc', 'agr': 'AgroVoc'}

for tag in tags:

    print(f'Running NETL labeler for {tag}')

    # The file in csv format which contains the topic terms that needs a label.
    data = f"{data_folder}/tok_{tag}.csv"
    # The output file for supervised labels.
    out_sup = f"{data_folder}/lab_{tag}.csv"

    print(f"-- Taxonomy: {taxonomy_name[taxonomy]}")
    print(f"-- Input data file: {data}")
    print(f"-- No. of supervised labels: {num_sup_labels}")

    if taxonomy == 'esv':
        sl = my_supervised_labels.SupLabeler()
    else:
        sl = my_supervised_labels.AGRLabeler()
    sl.get_labels(
        num_sup_labels, pagerank_model, data, path2tax, svm_classify,
        pretrained_svm_model, out_sup, load_map=True, p2wikifile=path2wikifile)

    print(f"-- Output written in {out_sup}")


print('-- End.')
