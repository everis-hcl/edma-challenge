"""
Thic code is an adaptation of script supervised_labels.py an drelated code, taken from the NETL library. with apache license 2.0.

Though the code has some major modifications with respect to the original
source (class structuring, performance improvements), the preocessing steps
are the same.

The original code is at
(https://github.com/sb1992/NETL-Automatic-Topic-Labelling-)

Author:         Shraey Bhatia
Date:           October 2016
File: 		supervised_labels.py

Updated by:     Sihwa Park
Date:           January 7, 2019
Fix:            Updated to work with Python 3.6.5

This python code gives the top supervised labels for that topic. The paramters
needed are passed through get_labels.py.

It generates letter_trigram,pagerank, Topic overlap and num of words in
features. Then puts it into SVM classify format and finally uses the already
rained supervised model to make ranking predictions and get the best label.
You will need SVM classify binary from SVM rank. The URL is provided in readme.
"""

import pandas as pd
import numpy as np
import pathlib
import re
from scipy.spatial.distance import cosine
from collections import defaultdict, Counter
import os
import sys
import wikipedia
import yaml


# Employed to change the format of features.
def change_format(f1):
    lt_dict = defaultdict(dict)

    for elem in f1:
        x, y, z = elem
        lt_dict[z][x] = y
    return lt_dict


def chunks(k, n):
    n = max(1, n)
    return [k[i:i + n] for i in range(0, len(k), n)]


def norm(x):
    """
    Transform string x into the normalized form used in pare_rank_dict
    """
    return x.lower().replace(' ', '_')


class SupLabeler(object):

    def __init__(self, p2wikimap=None):
        """
        A classifier of list of tokens into categories, based on the NETL
        labeller

        Parameters
        ----------
        p2wikimap: str or None, optional (default=None)
            Path to the yaml file mapping candidate labels into the labels used
            by the pagerank_model (which are based on wikipedia titles).
            If None, a default path is used
        """

        self.tax2wiki = None
        self.cats_out = None
        self.wiki_titles = None

        # Path to the fila that contains (or will contain) the map from
        # target categories to wikipedia titles.
        self.p2wikimap = pathlib.Path('cats') / 'map_esv2wiki.yml'

        return

    def load_wikifile(self, path2wikifile):

        print('-- -- Loading wikipedia file...')
        with open(path2wikifile, 'r', encoding='utf-8') as f:
            text = f.readlines()

        text = [norm(x[:-1]) for x in text if len(x) > 1]

        # text = text[40:-12]
        # text = [x[31:-1] for x in text]
        # text = '),('.join(text)
        # text = text.split('),(')
        # text = [x.split(',')[2][1:-1] for x in text]
        # Remove repetitions
        # text = sorted(list(set(text)))

        return text

    def get_topic_lt(self, elem):
        """
        Get a letter trigram for topic terms in elem
        """

        tot_list = []
        for item in elem:
            trigrams = [item[i: i + 3] for i in range(0, len(item) - 2)]
            tot_list = tot_list + trigrams
        x = Counter(tot_list)
        total = sum(x.values(), 0.0)
        for key in x:
            x[key] /= total
        return x

    def get_lt_ranks(self, topic_list, lab_list, num):
        """
        This method will be used to get first feature of letter trigrams for
        candidate labels and then rank them. It uses cosine similarity to get a
        score between a letter trigram vector of label candidate and vector of
        topic terms.

        The ranks are given based on that score.
        """

        # Will get letter trigram for topic terms.
        topic_ls = self.get_topic_lt(topic_list[num])
        val_list = []
        final_list = []

        for item in lab_list:
            # get the trigrams for label candidate.
            trigrams = [item[i: i + 3] for i in range(0, len(item) - 2)]

            label_cnt = Counter(trigrams)
            total = sum(label_cnt.values(), 0.0)

            tot_keys = list(set(
                list(topic_ls.keys()) + list(label_cnt.keys())))
            listtopic = [topic_ls[elem] if elem in topic_ls else 0.0
                         for elem in tot_keys]
            listlabel = [label_cnt[elem] / total if elem in label_cnt else 0.0
                         for elem in tot_keys]

            # Cosine similarity.
            val = 1 - cosine(np.array(listtopic), np.array(listlabel))
            val_list.append((item, val))

        rank_val = [i[1] for i in val_list]
        arr = np.array(rank_val)
        order = arr.argsort()
        ranks = order.argsort()

        final_list = [(elem[0], ranks[i], int(num))
                      for i, elem in enumerate(val_list)]

        return final_list

    def read_candidates(self, path2tax):

        # Read taxonomy
        print('-- Loading taxonomy...')
        tax = pd.read_excel(
            path2tax, header=0, names=None, index_col=None, usecols=None,
            squeeze=False, dtype=None, converters=None, true_values=None,
            false_values=None, skiprows=None, nrows=None, na_values=None,
            keep_default_na=True)
        print(f'-- Extracting category list...')
        # Move all categories to a single list
        candidates = tax['Category Label'].tolist()

        return candidates

    def wikitest(self, cat, page_rank_dict):
        """
        Tests if a string has an equivalent entry in the wikipedia
        If so, the correpondence is added to the self.tax2map mapping.
        If not, the category is returned as the unique elemente of a string,
        to be easyly processed
        """

        # By default, the category is taken as unmatched to any wiki category.
        out = [cat]
        if (self.wiki_titles is None) or (norm(cat) in self.wiki_titles):
            try:
                # Get the page title in the normalized form used in
                # page_rank_dict
                wiki_token = norm(wikipedia.page(cat).title)
                if wiki_token in page_rank_dict:
                    if wiki_token not in self.tax2wiki.values():
                        self.tax2wiki[cat] = wiki_token
                        out = []
                    else:
                        print(f"Category {cat} cannot be mapped to "
                              f"{wiki_token} because it already exists")
            except:
                if ((self.wiki_titles is not None)
                        and (cat in self.wiki_titles)):
                    print(f'-- -- Warning: {cat} is in wiki_titles, but not '
                          'in the web')

        return out

    def get_map2wiki(self, cats, page_rank_dict):
        """
        Returns a dictionary mapping categories from a given list into
        categories from the wikipedia.

        The mapping is imperfect. Unmatched tokens are returned in a separate
        list

        Parameters
        ----------
        cats : list
            Input categories.
        page_rank_dict : list of dict
            List of target categories. If dict, the list is given by the
            dictionary entries

        Returns
        -------
        tax2wiki : dict
            A map from input categories to wikipedia entries
        tokens_out : list
            A list of input categories that could not be mapped to wikipedia
            entries

        Note
        ----
        This code is based on a list of wikipedia entries that may be currenly
        obsolete. Some wikipedia entries might not exist in the current
        wikipedia
        """

        # Replace spaces in categories by underscore
        self.cats_out = sorted(cats)

        # This will contain the output dictionary
        self.tax2wiki = {}
        # This will contain the unmatched input categories
        print(f'-- Input list with {len(self.cats_out)} categories')

        # ################################
        # Exact matches to wiki categories

        # Map categories to themselves (in lowercase), if possible
        # Lowercase is applied becasue page_rank_dict entres do not contain
        # capital letters
        matched_cats = [x for x in self.cats_out if norm(x) in page_rank_dict]
        self.cats_out = sorted(list(set(self.cats_out) - set(matched_cats)))
        self.tax2wiki.update({x: norm(x) for x in matched_cats})
        print(f"-- -- {len(matched_cats)} categories mapped to themselves")

        # Map category names in plural to a singular version based on removing
        # the last s, if possible
        matched_cats = [x for x in self.cats_out
                        if norm(x)[:-1] in page_rank_dict]
        self.cats_out = sorted(list(set(self.cats_out) - set(matched_cats)))
        self.tax2wiki.update({x: norm(x)[:-1] for x in matched_cats})
        print(f"-- -- {len(matched_cats)} categories mapped to a singular "
              "version")
        print(matched_cats)

        # Map category names in plural to a singular version based on removing
        # the last s, if possible
        matched_cats = [x for x in self.cats_out if norm(x)[-3:] == 'ies'
                        and norm(x)[:-3] + 'y' in page_rank_dict]
        self.cats_out = sorted(list(set(self.cats_out) - set(matched_cats)))
        self.tax2wiki.update({x: norm(x)[:-3] + 'y' for x in matched_cats})
        print(f"-- -- {len(matched_cats)} categories mapped to a singular "
              "version")
        print(matched_cats)

        # #################################
        # Exact matches to redirected-pages

        # Map categories to redirected pages
        new_out = []
        for i, cat in enumerate(self.cats_out):
            print(f"Category {i+1} out of {len(self.cats_out)}    \r", end='')
            out = self.wikitest(cat, page_rank_dict)
            new_out += out

        n_matches = len(self.cats_out) - len(new_out)
        print(f"-- -- {n_matches} categories mapped to redirected pages")
        self.cats_out = sorted(new_out)

        # Map categories to redirected pages from the singular version.
        new_out = []
        for i, cat in enumerate(self.cats_out):
            print(f"Category {i+1} out of {len(self.cats_out)}    \r", end='')
            # Select plurals only
            if cat[-1] == 's':
                # Rude conversion to singular form...
                cat0 = cat[: -1]
                out = self.wikitest(cat0, page_rank_dict)
                new_out += out
            else:
                new_out.append(cat)

        n_matches = len(self.cats_out) - len(new_out)
        print(f"-- -- {n_matches} categories in singular mapped to redirected "
              "pages")
        self.cats_out = sorted(new_out)

        # This is an additional manual map.
        cats2wiki = {
            'apidology': 'melittology',
            'autonomous_vehicle': 'autonomous_car',
            'carbon_fiber': 'carbon_fibers',
            'climatic_changes': 'climate_change',
            'closed-loop_systems': 'closed_loop',
            'continuous_glucose_monitors': 'blood_glucose_monitoring',
            'diabetes': 'diabetes_mellitus',
            'ebola': 'ebola_virus_disease',
            'educational_sciences': 'education_theory',
            'el_niÃ±o': 'el_niño',
            'electric_batteries': 'battery_(electricity)',
            'electrical_engineering,_electronic_engineering,_information_engineering':
                'electrical_engineering',
            'experimental_petrology': 'petrology',
            'fiber-optic_network': 'fiber-optic_communication',
            'inclusive_education': 'inclusion_(education)',
            'modeling_of_disease_spread':
                'mathematical_modelling_of_infectious_disease',
            'social_aspects_of_transport': 'mobility',
            'soft_robotics': 'robotic_process_automation',
            'squamous_cell_carcinoma': 'squamous-cell_carcinoma',
            'transplantation': 'transplant'}
        # Filter out entries whose image does not match page_rank_dict
        # categories (there should be no one)
        cats2wiki = {x: y for x, y in cats2wiki.items() if y in page_rank_dict
                     and y not in self.tax2wiki.values()}
        self.cats_out = sorted(list(set(self.cats_out) - set(cats2wiki)))
        # Add new entries to the map
        self.tax2wiki.update(cats2wiki)

        print(f"-- -- {len(self.tax2wiki)} categories mapped")
        print(f"-- -- {len(self.cats_out)} unmatched categories: "
              f"{self.cats_out}")

        return

    def prepare_features(self, letter_tg_dict, page_rank_dict, cols,
                         feature_names, topic_list, categories):
        """
        This method is to prepare all features. It will take in dictionary of
        letter trigram, pagerank, list of all columns for the dataframe and
        name of features. It will generate four features in the dataframe
        namely Pagerank, letter trigram, Topic overlap and Number of words in
        a label. Additionally DatFrame will also be given the label name,
        topic_id and an avg_val which is average annotator value. It is just
        given a value of 3 here but can be anything as it does not make a
        difference in prediction. Only important when we have to train SVM
        model.

        Parameters
        ----------
        letter_tg_dict: dict
            Dict
        page_rank_dict: dict
            Dict
        cols:
            cols
        feature_names:
            cols
        topic_list: list
            Topics
        categories: dict
            List of candidate labels (based on wikipedia titles)
        """

        frame = pd.DataFrame()

        for x in range(0, len(letter_tg_dict)):
            print(f"-- -- Item {x+1} out of {len(letter_tg_dict)}         \r",
                  end="")
            a = letter_tg_dict[x]
            temp_list = []

            for k in a:

                if k in categories:
                    pagerank = float(page_rank_dict[k])
                else:
                    pagerank = np.nan

                # Extracting topic overlap.
                word_labels = k.split("_")
                com_word_length = len(
                    set(word_labels).intersection(set(topic_list[x])))

                # number of words in the candidate label.
                lab_length = len(word_labels)

                # The list created to get values for dataframe
                # [label name, topic id, letter trigram, pagerank value,
                # no. of words, topic overlap, arbitrary value]
                # The arbitrary value is for the sake of giving
                # column for annotator rating neeeded in SVM Ranker classify
                new_list = [k, x, a[k], pagerank, lab_length, com_word_length,
                            3]
                temp_list.append(new_list)

                # temp = pd.Series(new_list, index=cols)
                # temp_frame = temp_frame.append(temp, ignore_index=True)

            temp_frame = pd.DataFrame(temp_list, columns=cols)
            # Just filling in case a label does not have a pagerank value.
            # Generally should not happen
            temp_frame = temp_frame.fillna(0)

            for item in feature_names:
                # feature Normalization per topic.
                temp_frame[item] = (
                    (temp_frame[item] - temp_frame[item].mean())
                    / (temp_frame[item].max() - temp_frame[item].min()))
            frame = frame.append(temp_frame, ignore_index=True)
            frame = frame.fillna(0)
        return frame

    def convert_dataset(self, test_file, feature_names):
        """
        Converts the dataset into a format which is taken by SVM
        ranker classify binary file.
        """

        test_list = []
        for x in test_file.itertuples():
            mystring = (
                f"{x.avg_val} qid:{int(x.topic_id)}"
                + ''.join([f" {j + 1}:{x._asdict()[item]}"
                           for j, item in enumerate(feature_names)])
                + f" # {x.label}")
            test_list.append(mystring)

        return test_list

    def get_predictions(self, test_set, num, svm_classify, trained_svm_model,
                        num_sup_labels):
        """
        It calls SVM classify and gets predictions for each topic.
        """

        h = open("test_temp.dat", "w", encoding="utf-8")
        for item in test_set:
            h.write("%s\n" % item)
        h.close()

        query2 = (f"{svm_classify} test_temp.dat {trained_svm_model}"
                  + " predictionstemp")
        print(query2)
        os.system(query2)
        h = open("predictionstemp", encoding="utf-8")
        pred_list = []
        for line in h:
            pred_list.append(line.strip())
        h.close()

        pred_chunks = chunks(pred_list, num)
        test_chunks = chunks(test_set, num)
        list_max = []
        for j in range(len(pred_chunks)):
            max_sort = np.array(
                pred_chunks[j]).argsort()[::-1][:int(num_sup_labels)]
            list_max.append(max_sort)
        print("\n")
        print("Printing Labels for supervised model")
        # g = open(output_supervised, 'w', encoding="utf-8")
        # for cnt, (x, y) in enumerate(zip(test_chunks, list_max)):
        #     print(f"Top {num_sup_labels} labels for topic {cnt} are:")
        #     # g.write("Top "+ num_sup_labels+" labels for topic "
        #     #         + str(cnt)+" are:" +"\n")
        #     g.write(str(cnt) + ",")

        #     for index, i2 in enumerate(y):
        #         m = re.search('# (.*)', x[i2])
        #         cat = wiki2tax[m.group(1)]
        #         print(cat)
        #         g.write(cat)
        #         if index != (len(y) - 1):
        #             g.write(",")

        #     print("\n")
        #     g.write("\n")
        # g.close()

        labels = []
        for cnt, (x, y) in enumerate(zip(test_chunks, list_max)):

            labels_cnt = []
            for index, i2 in enumerate(y):
                m = re.search('# (.*)', x[i2])
                labels_cnt.append(m.group(1))
            labels.append(labels_cnt)

        # deleting the ttest file and prediction file generated as part of code
        # to run svm_classify
        query3 = "rm test_temp.dat predictionstemp"
        os.system(query3)

        return labels

    def get_labels(self, num_sup_labels, pagerank_model, data,
                   path2tax, svm_classify, trained_svm_model,
                   output_supervised, load_map=False, p2wikifile=None):
        """
        Parameters
        ----------
        num_sup_labels: int
            num of supervised labels needed.
        pagerank_model: str
            path to the pagerank file
        data: str
            path to the topic data file.
        output_candidates: str
            path of generated candidate file.
        svm_classify: str
            path to the SVM Ranker classify binary file. Needs to be downloaded
            from the path provided in Readme.
        trained_svm_model:
            This is the pre existing trained SVM model, trained on our SVM
            model.
        path2tax: str
            Output file for supervised labels
        load_map: bool, optional (default=False)
            If True, the label map is loaded from path2wikimap (if provided)
            If False, the label map is computed, and saved into path2wikimap
            (if provided)
        """

        # This is for printing purposes only
        n_steps = 8

        # Load the pagerank File into a dictionary
        # Entries of p_rank_dict are tokens, values are float.
        # The tokens are wikipedia titles.
        print(f"-- STEP 1/{n_steps}: Loading pageRank models...")
        with open(pagerank_model, 'r', encoding="utf-8") as f:
            pr_model_raw = f.readlines()
        pr_model_raw = [x.split() for x in pr_model_raw]
        p_rank_dict = {x[1].lower(): x[0] for x in pr_model_raw}

        # Get map candidate labels --> page_rank labels
        print(f"-- STEP 2/{n_steps}: Loading taxonomy-wikipedia map")
        if load_map and os.path.isfile(self.p2wikimap):
            # Load mapping from yaml file
            with open(self.p2wikimap, 'r', encoding="utf-8") as f:
                self.tax2wiki = yaml.full_load(f)
            print('-- Map to wiki labels loaded')
        else:

            print(f"-- -- Wikipedia map is not available in "
                  f"{self.p2wikimap}. Computing map. This may take a while...")

            # Load all wikipedia titles from file (only if a path is provided
            # in p2wikifile. The titles are stored in self.wiki_titles and
            # will be used by method self.wikitest inside self.get_map2wiki)
            if p2wikifile is not None:
                self.wiki_titles = self.load_wikifile(p2wikifile)

            # Read target categories
            target_labels = self.read_candidates(path2tax)

            self.get_map2wiki(target_labels, p_rank_dict)

            # Save categories to file
            with open(self.p2wikimap, 'w', encoding="utf-8") as f:
                yaml.dump(self.tax2wiki, f, default_flow_style=False,
                          sort_keys=True)

        wiki_targets = list(self.tax2wiki.values())

        # Inverse map
        wiki2tax = {y: x for x, y in self.tax2wiki.items()}

        # Select a 1-1 map from self.tax2wiki
        if len(set(wiki_targets)) < len(self.tax2wiki):
            print("-- ERROR: the taxonomy map is not one-to-one")
            breakpoint()
            exit()

        # Just get the number of labels per topic.
        # Note that we assume that, for each in put, the number of target
        # labels is the same
        # test_chunk_size = len(target_labels[0])
        test_chunk_size = len(wiki_targets)

        # Number of Supervised labels needed should not be less than the
        # number of candidate labels.
        if test_chunk_size < int(num_sup_labels):
            print("\nError: You cannot extract more labels than present in "
                  "input file")
            sys.exit()

        # Reading in the topic terms from the topics file.
        # These are the inputs to be classified/labeled
        print(f"-- STEP 3/{n_steps}: Loading inputs (lists of terms to be "
              "classified")
        topics = pd.read_csv(data)
        try:
            new_frame = topics.drop('domain', 1)
            topic_list = new_frame.set_index('topic_id').T.to_dict('list')
        except:
            topic_list = topics.set_index('topic_id').T.to_dict('list')

        # This calls the above method to get letter trigram feature.
        print(f"-- STEP 4/{n_steps}: Computing letter trigram ranks")
        temp_lt = []
        for j in range(0, len(topic_list)):
            print(f"-- -- Computing lt_ranks for topic {j + 1} out of "
                  f"{len(topic_list)}                       \r", end="")
            temp_lt.append(self.get_lt_ranks(topic_list, wiki_targets, j))

        letter_trigram_feature = [item for sublist in temp_lt
                                  for item in sublist]
        print("-- Reformating letter trigram feature                      ")
        lt_dict = change_format(letter_trigram_feature)

        print(f"-- STEP 5/{n_steps}: Preparing features:")
        # Name of columns for DataFrame.
        cols = ['label', 'topic_id', 'letter_trigram', 'prank', 'lab_length',
                'common_words', 'avg_val']
        # Name of features.
        features = ['letter_trigram', 'prank', 'lab_length', 'common_words']
        feature_dataset = self.prepare_features(
            lt_dict, p_rank_dict, cols, features, topic_list, wiki_targets)
        print("-- All features generated                                ")

        # Convert the dataset into a format which is taken by SVM ranker
        # classify binary file.
        print(f"-- STEP 6/{n_steps}: Dataset conversion")
        test_list = self.convert_dataset(feature_dataset, features)

        # It calls SVM classify and gets predictions for each topic.
        print(f"-- STEP 7/{n_steps}: Computing predictions")
        labels = self.get_predictions(
            test_list, test_chunk_size, svm_classify, trained_svm_model,
            num_sup_labels)
        print("-- Predictions computed")

        # Convert wiki-labels to target lagels:
        print(f"-- STEP 8/{n_steps}: Saving results")
        labels = [[wiki2tax[x] for x in y] for y in labels]
        df_labels = pd.DataFrame(labels)
        df_labels.to_csv(output_supervised)

        print(labels)

        return


class AGRLabeler(SupLabeler):

    def __init__(self):

        super().__init__()

        # Path to the fila that contains (or will contain) the map from
        # target categories to wikipedia titles.
        self.p2wikimap = pathlib.Path('cats') / 'map_agr2wiki.yml'

        return

    def read_candidates(self, path2tax):

        # Read taxonomy
        print('-- Loading taxonomy (this may take a while)...')
        with open(path2tax, 'r', encoding="utf-8") as f:
            candidates = yaml.full_load(f)

        return candidates

    def get_map2wiki(self, cats, page_rank_dict):
        """
        Returns a dictionary mapping categories from a given list into
        categories from the wikipedia.

        The mapping is imperfect. Unmatched tokens are returned in a separate
        list

        Parameters
        ----------
        cats : list
            Input categories.
        page_rank_dict : list of dict
            List of target categories. If dict, the list is given by the
            dictionary entries

        Note
        ----
        This code is based on a list of wikipedia entries that may be currenly
        obsolete. Some wikipedia entries might not exist in the current
        wikipedia
        """

        # Map concepts to a single label, taken from the literalForm or the
        # prefLabel, and then take the inverse map
        # tag2concept = {x: y['literalForm'] + y['prefLabel']
        #                for x, y in cats.items()}
        tag2concept = {x: y['prefLabel'] for x, y in cats.items()}
        # Inverse map
        tag2concept = {y: x for x, y in tag2concept.items()}

        # Save inverse map into file
        fpath = pathlib.Path('cats') / 'tag2concept.yml'
        with open(fpath, 'w', encoding="utf-8") as f:
            yaml.dump(tag2concept, f, default_flow_style=False, sort_keys=True)

        # Select list of categories
        # self.cats_out = [(y['literalForm'] + y['prefLabel'], y['altLabel'])
        #                  for y in cats.values()]
        # self.cats_out = sorted(list(set([y['literalForm'] + y['prefLabel']
        #                                  for y in cats.values()])))
        self.cats_out = sorted(list(set([
            y['prefLabel'] for y in cats.values()])))

        # This will contain the output dictionary
        self.tax2wiki = {}
        # This will contain the unmatched input categories
        print(f'-- Input list with {len(self.cats_out)} categories')

        # ################################
        # Exact matches to wiki categories

        # Map categories to themselves (in lowercase), if possible
        # Lowercase is applied becasue page_rank_dict entres do not contain
        # capital letters
        matched_cats = [x for x in self.cats_out if norm(x) in page_rank_dict]
        self.cats_out = sorted(list(set(self.cats_out) - set(matched_cats)))
        self.tax2wiki.update({x: norm(x) for x in matched_cats})
        print(f"-- -- {len(matched_cats)} categories mapped to themselves")

        # Map category names in plural to a singular version based on removing
        # the last s, if possible
        matched_cats = [x for x in self.cats_out
                        if norm(x)[:-1] in page_rank_dict]
        self.cats_out = sorted(list(set(self.cats_out) - set(matched_cats)))
        self.tax2wiki.update({x: norm(x)[:-1] for x in matched_cats})
        print(f"-- -- {len(matched_cats)} categories mapped to a singular "
              "version")
        print(matched_cats)

        # Map category names in plural to a singular version based on removing
        # the last s, if possible
        matched_cats = [x for x in self.cats_out if x.lower()[-3:] == 'ies'
                        and norm(x)[:-3] + 'y' in page_rank_dict]
        self.cats_out = sorted(list(set(self.cats_out) - set(matched_cats)))
        self.tax2wiki.update({x: norm(x)[:-3] + 'y' for x in matched_cats})
        print(f"-- -- {len(matched_cats)} categories mapped to a singular "
              "version")
        print(matched_cats)

        # #################################
        # Exact matches to redirected-pages

        # Map categories to redirected pages
        new_out = []
        for i, cat in enumerate(self.cats_out):
            print(f"Category {i+1} out of {len(self.cats_out)}    \r", end='')
            out = self.wikitest(cat, page_rank_dict)
            new_out += out

        n_matches = len(self.cats_out) - len(new_out)
        print(f"-- -- {n_matches} categories mapped to redirected pages")
        self.cats_out = sorted(new_out)

        # Map categories to redirected pages from the singular version.
        new_out = []
        for i, cat in enumerate(self.cats_out):
            print(f"Category {i+1} out of {len(self.cats_out)}    \r", end='')
            # Select plurals only
            if cat[-1] == 's':
                # Rude conversion to singular form...
                cat0 = cat[: -1]
                out = self.wikitest(cat0, page_rank_dict)
                new_out += out
            else:
                new_out.append(cat)

        n_matches = len(self.cats_out) - len(new_out)
        print(f"-- -- {n_matches} sing. categories mapped to redirected pages")
        self.cats_out = sorted(new_out)

        # #############################
        # Filter out duplicated targets
        # A few cases may arise where the targets come from different labels
        # but with the same normalized form. To make sure that the map is one-
        # to-one, we make a double dictionary inversion
        all_cats = set(self.tax2wiki.keys())
        self.tax2wiki = {y: x for x, y in self.tax2wiki.items()}
        self.tax2wiki = {y: x for x, y in self.tax2wiki.items()}
        new_out = list(all_cats - set(self.tax2wiki.keys()))
        self.cats_out = sorted(self.cats_out + new_out)

        # ####################################
        # Exact matches using alternate labels

        # Discard categories with no alternate label
        cats_out_def = [x for x in self.cats_out if len(x[1]) == 0]
        cats_2_test = [x for x in self.cats_out if len(x[1]) > 0]

        print('-- -- SUMMARY: ')
        print(f"-- -- {len(self.tax2wiki)} categories mapped")
        print(f"-- -- {len(self.cats_out)} unmatched categories: ")
        print(f"-- -- {len(cats_out_def)} cats definitely unmatched")
        print(f"-- -- {len(cats_2_test)} cats remain with alternate labels")

        return

