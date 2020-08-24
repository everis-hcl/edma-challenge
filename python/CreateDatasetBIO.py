import requests
import pandas as pd
from pathlib import Path
from io import StringIO
import json
import time
from collections import Counter
# import matplotlib.pyplot as plt
import numpy as np

import ipdb

from utils import printgr, printred, printmag
# from dbManager.base_dm_sql import BaseDMsql

#Conectamos a la Base de Datos de Semantic Scholar
dbCONNECTOR = 'mysql'
dbNAME = 'db_Pu_S2'
dbUSER = 'PTLprojects'
dbPASS = 'Kts93_u17a'
dbSERVER = 'localhost'
dbSOCKET = '/var/run/mysqld/mysqld.sock'
# DM = BaseDMsql(db_name=dbNAME, db_connector=dbCONNECTOR, path2db=None,
#                db_server=dbSERVER, db_user=dbUSER, db_password=dbPASS,
#                unix_socket=dbSOCKET)

def get_S2_data(data_folder, csv_file, S2_folder):
    """
    Download Semantic Scholar description of all papers and authors in base dataset
    :param data_folder: Path to folder with the Bio-protocol .json files
    :param csv_file: Path to csv file with PMID to PMC translations
    :param S2_folder: Path to folder where the downloaded papers will be saved
    """

    paperUrl = 'https://api.semanticscholar.org/v1/paper/XXXXX'
    authorUrl = 'https://api.semanticscholar.org/v1/author/XXXXX'
    
    if not S2_folder.exists():
        S2_folder.mkdir()

    papers = [f for f in data_folder.iterdir() if f.name.endswith('.json')]

    ProtocolID = []
    S2ID = []

    for idx,el in enumerate(papers):
        time.sleep(5)
        doi = '10.21769/BIOPROTOC.' + el.name.split('Bio-protocol')[1].split('.json')[0]
        printgr('Processing paper ' + el.name + ' (' + str(idx) + '). DOI: ' + doi)
        response = requests.get(url = paperUrl.replace('XXXXX', doi))
        if not response.ok:
            print(response.status_code)
        if response.status_code==404:
            time.sleep(5)
            response = requests.get(url = paperUrl.replace('XXXXX', doi.lower()))
        if response.ok:
            paperdata = json.load(StringIO(response.text))
            with S2_folder.joinpath(paperdata['paperId'] +'.json').open('w') as fout:
                json.dump(paperdata, fout)

            for author in paperdata['authors']:
                authorId = author['authorId']
                time.sleep(5)
                response = requests.get(url = authorUrl.replace('XXXXX', str(authorId)))
                if not response.ok:
                    print(response)
                    print(authorId)
                else:
                    authordata = json.load(StringIO(response.text))
                    #Save author data
                    with S2_folder.joinpath('Author' + str(authorId) +'.json').open('w') as fout:
                        json.dump(authordata,fout)

            ProtocolID.append(el.name.split('.json')[0])
            S2ID.append(paperdata['paperId'])

    with csv_file.open('w') as fout:
        fout.write('ProtocolID,S2ID\n')
        [fout.write(PID+','+S2ID+'\n') for PID, S2ID in zip(ProtocolID,S2ID)]
    return


# def extendBIO(S2_folder, csv_file_extended):
#     """
#     Analyze paper distribution according to year and field of Study
#     And generate extended dataset applying filters based on this metadata
#     :param S2_folder: Path to folder where S2 paper and author information is available
#     :param csv_file_extended: Path to file where the identifiers of the extended dataset
#                               will be saved
#     """
    
#     # Generamos las listas de identificadores S2 de los artículos en los datasets ampliados
#     author_files = sorted([d for d in S2_folder.iterdir() if d.name.startswith('Author') and d.name.endswith('.json')])
#     print('Número de autores en el dataset base:', len(author_files))
#     author_counts = []
#     author_S2 = []
#     author_PMID = []
#     author_Abstract = []
#     for af in author_files:
#         with af.open() as fin:
#             author_data = json.load(fin)
#         author_counts.append(len(author_data['papers']))
#         for paper in author_data['papers']:
#             author_S2.append(paper['paperId'])
#             df = DM.readDBtable('S2papers', limit=None, selectOptions='pmid, paperAbstract, doi',
#                                 filterOptions='S2paperID="'+paper['paperId']+'"')
#             if len(df):
#                 pmid = df.values.tolist()[0][0]
#                 abstract = df.values.tolist()[0][1]
#                 if len(pmid):
#                     author_PMID.append(paper['paperId'])
#                 if len(abstract):
#                     author_Abstract.append(paper['paperId'])
#     author_S2 = set(author_S2)
#     author_Abstract = set(author_Abstract)
#     author_PMID = set(author_PMID)
#     author_selection = [el for el in list(author_PMID) if el in author_Abstract]
#     print('Número total de artículos asociados a los autores:', len(author_S2))
#     print('Número total de artículos asociados con un PMID válido', len(author_PMID))
#     print('Número total de artículos asociados con Abstract', len(author_Abstract))
#     print('Número total de artículos con Abstract y PMID válido', len(author_selection))

#     paper_files = sorted([d for d in S2_folder.iterdir() if (not d.name.startswith('Author')) and d.name.endswith('.json')])
#     print('Número de papers en el dataset base:', len(paper_files))
#     reference_counts = []
#     reference_S2 = []
#     reference_PMID = []
#     reference_Abstract = []
#     for pf in paper_files:
#         with pf.open() as fin:
#             paper_data = json.load(fin)
#         reference_counts.append(len(paper_data['references']))
#         for paper in paper_data['references']:
#             reference_S2.append(paper['paperId'])
#             df = DM.readDBtable('S2papers', limit=None, selectOptions='pmid, paperAbstract',
#                                 filterOptions='S2paperID="'+paper['paperId']+'"')
#             if len(df):
#                 pmid = df.values.tolist()[0][0]
#                 abstract = df.values.tolist()[0][1]
#                 if len(pmid):
#                     reference_PMID.append(paper['paperId'])
#                 if len(abstract):
#                     reference_Abstract.append(paper['paperId'])

#     reference_S2 = set(reference_S2)
#     reference_Abstract = set(reference_Abstract)
#     reference_PMID = set(reference_PMID)
#     reference_selection = [el for el in list(reference_PMID) if el in reference_Abstract]
#     print('Número total de referencias asociadas al dataset base:', len(reference_S2))
#     print('Número total de referencias asociadas con un PMID válido', len(reference_PMID))
#     print('Número total de referencias asociadas con Abstract', len(reference_Abstract))
#     print('Número total de referencias con Abstract y PMID válido', len(reference_selection))

#     print('##########')
#     selected = set(list(author_selection)+list(reference_selection))
#     selected_abstract = set(list(author_Abstract)+list(reference_Abstract))
#     print('Número total de papers con Abstract y PMID:', len(selected))
#     print('Número total de papers con Abstract:', len(selected_abstract))

#     base_papers = set([el.name.split('.json')[0] for el in paper_files])
#     extended = set([el for el in list(selected) if el not in base_papers])
#     extended_abstract = set([el for el in list(selected_abstract) if el not in base_papers])
#     printgr('Estudio estadístico del corpus de datos extendido')
#     print('Número de papers en el dataset base:', len(base_papers))
#     print('Número adicional de papers en el conjunto extendido:', len(extended))
#     print('Número adicional de papers en el conjunto extendido (abstract):', len(extended_abstract))

#     #Year distribution y filtramos los papers con fechas anteriores a 2000 o no definidas
#     df = pd.DataFrame()
#     for paper in list(selected_abstract):
#         dfaux = DM.readDBtable('S2papers', limit=None, selectOptions='S2paperID, doi, year, fieldsOfStudy',
#                                 filterOptions='S2paperID="'+paper+'"')
#         df = df.append(dfaux, ignore_index = True)
#     years_base = [el[2] for el in df.values.tolist() if el[0] in base_papers]
#     years_extended = [el[2] for el in df.values.tolist() if el[0] in extended]
#     years_abstract = [el[2] for el in df.values.tolist() if el[0] in extended_abstract]
#     printgr('Distribución de años para dataset base')
#     print(Counter(years_base))
#     printgr('Distribución de años para dataset extendido')
#     print(Counter(years_extended))
#     printgr('Distribución de años para dataset extendido (solo abstract)')
#     print(Counter(years_abstract))

#     #Filtering papers by years
#     df = df.loc[(df['year'] >= 2000) & (df['year'] <= 2020)]

#     #fieldsOfStudy distribution
#     fields_base = [el[3] for el in df.values.tolist() if el[0] in base_papers]
#     fields_extended = [el[3] for el in df.values.tolist() if el[0] in extended]
#     fields_abstract = [el[3] for el in df.values.tolist() if el[0] in extended_abstract]
#     fields_base = [el.split('\t') for el in fields_base]                                                                                     
#     fields_extended = [el.split('\t') for el in fields_extended]                                                                             
#     fields_abstract = [el.split('\t') for el in fields_abstract]
#     fields_base = [item for sublist in fields_base for item in sublist]
#     fields_extended = [item for sublist in fields_extended for item in sublist]
#     fields_abstract = [item for sublist in fields_abstract for item in sublist]
#     printgr('Distribución de fieldsOfStudy para dataset base')
#     print(Counter(fields_base))
#     printgr('Distribución de fieldsOfStudy para dataset extendido')
#     print(Counter(fields_extended))
#     printgr('Distribución de fieldsOfStudy para dataset extendido (solo abstract)')
#     print(Counter(fields_abstract))

#     """#Filtering by fieldsOfStudy
#     df = df.loc[(df['fieldsOfStudy'].str.contains('Medicine')) | (df['fieldsOfStudy'].str.contains('Biology')) | (df['fieldsOfStudy'].str.contains('Chemistry')) ]
    
#     #New fieldsOfStudy distribution (after filtering Medicine / Biology / Chemistry papers
#     fields_base = [el[3] for el in df.values.tolist() if el[0] in base_papers]
#     fields_extended = [el[3] for el in df.values.tolist() if el[0] in extended]
#     fields_abstract = [el[3] for el in df.values.tolist() if el[0] in extended_abstract]
#     fields_base = [el.split('\t') for el in fields_base]
#     fields_extended = [el.split('\t') for el in fields_extended]
#     fields_abstract = [el.split('\t') for el in fields_abstract]
#     fields_base = [item for sublist in fields_base for item in sublist]
#     fields_extended = [item for sublist in fields_extended for item in sublist]
#     fields_abstract = [item for sublist in fields_abstract for item in sublist]
#     printgr('Distribución de fieldsOfStudy para dataset base')
#     print(Counter(fields_base))
#     printgr('Distribución de fieldsOfStudy para dataset extendido')
#     print(Counter(fields_extended))
#     printgr('Distribución de fieldsOfStudy para dataset extendido (solo abstract)')
#     print(Counter(fields_abstract))
#     """

#     #Save identifiers of extended datasets
#     df.to_csv(csv_file_extended, index=False)
    
#     #Obtenemos distribución de número de papers por autor
#     selected = set([el[0] for el in df.values.tolist()])
#     author_counts = []
#     for af in author_files:
#         with af.open() as fin:
#             author_data = json.load(fin)
#         author_counts.append(sum([1 for el in author_data['papers'] if el['paperId'] in selected]))
#     ipdb.set_trace()
#     hist, bins, _ = plt.hist(author_counts, bins=16)
#     logbins = np.logspace(np.log10(1),np.log10(bins[-1]),len(bins))
#     print(plt.hist(author_counts, bins=logbins))

#     return



if __name__ == "__main__":
    
    Path2data = Path('./data/BIO')
    data_folder = Path2data.joinpath('Bio-protocol')
    csv_file = Path2data.joinpath('BIO_S2.csv')
    S2_folder = Path2data.joinpath('Extended_BIO')
    csv_file_extended = Path2data.joinpath('S2ID_extendedBIO.csv')

    ######################################################
    # Download papers and authors information from S2
    # Create Bio-protocol to S2 dictionary
    ######################################################
    #get_S2_data(data_folder=data_folder, csv_file=csv_file, S2_folder=S2_folder)

    ######################################################
    # Generate extended dataset and analyze paper distributions
    ######################################################
    # extendBIO(S2_folder=S2_folder, csv_file_extended=csv_file_extended)

