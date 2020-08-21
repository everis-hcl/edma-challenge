"""
Reads an NT file conaining the AgroVoc taxonomy and returns a dictionary of
all concepts that have an associated english label.

The NT file is required. It can be downloaded from
http://agrovoc.uniroma2.it/agrovocReleases/agrovoc_2020-07-08_lod.nt.zip

The output dictionary is stored into a yaml file.

v1.0: Jesus Cid, Aug, 2020
"""

from rdflib.graph import Graph
import yaml

# #######################
# Configurable parameters

# Path to the NT file
fpath_in = "../source_data/taxonomy/agrovoc_2020-07-08_lod.nt"
# Path to the output file
fpath_out = '../agrovocab.yml'

# ###################################
# Extract AgroVoc concepts and labels
g = Graph()

print('-- Loading data...')
g.parse(fpath_in, format="nt")

terms = {}
print('-- Extracting terms...')
breakpoint()
for s, p, o in g:

    concept = s.split('/')[-1]

    # Select terms in english only
    if 'language' in dir(o) and o.language == 'en':

        # if concept not in terms:
        #     terms[concept] = {
        #         'literalForm': '', 'prefLabel': '', 'altLabel': []}

        # if p.endswith('literalForm'):
        #     terms[concept]['literalForm'] = o.title()
        # elif p.endswith('prefLabel'):
        #     terms[concept]['prefLabel'] = o.title()
        # elif p.endswith('altLabel'):
        #     terms[concept]['altLabel'].append(o.title())

        # if concept not in terms:
        #     terms[concept] = {
        #         'literalForm': '', 'prefLabel': '', 'altLabel': []}

        if not p.endswith('literalForm'):
            if concept not in terms:
                terms[concept] = {'prefLabel': '', 'altLabel': []}
            if p.endswith('prefLabel'):
                terms[concept]['prefLabel'] = o.title()
            elif p.endswith('altLabel'):
                terms[concept]['altLabel'].append(o.title())

# Filter out concepts without preferred label
# terms = {x: y for x, y in terms.items()
#          if y['literalForm'] != '' or y['prefLabel'] != ''}
# n_literal = len([x for x in terms.values() if x['literalForm'] != ''])
n_pref = len([x for x in terms.values() if x['prefLabel'] != ''])
n_alt = len([x for x in terms.values() if x['altLabel'] != []])
terms = {x: y for x, y in terms.items() if y['prefLabel'] != ''}
n_terms = len(terms)

print(f"-- {n_terms} selected")
# print(f"-- {n_literal} terms with literal form")
print(f"-- {n_pref} terms with pref label, selected")
print(f"-- {n_alt} terms with alt label")

# Write topics to csv file
with open(fpath_out, "w", encoding='utf-8') as f:
    yaml.dump(terms, f)

print(f"-- Taxonomy saved in {fpath_out}")

print(f'Taxonomi')
