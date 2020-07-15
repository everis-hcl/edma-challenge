import requests
import pandas as pd
from pathlib import Path
from io import StringIO
import json
import time

import ipdb

from utils import printgr, printred, printmag

def PMC_to_PMID_S2(PMClist_file, csv_file):
    """
    Translate PMC to PMID and S2 (Semantic Scholar)
    :param PMClist_file: Location of text file with full list of PMCs for dataset
    :param csv_file: Route where the CSV file with translations will be saved
    """

    #We translate PMC to PMID with the help with the following online service
    url = 'https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/'

    with PMClist_file.open() as fin:
        pmcs = [el.strip() for el in fin.readlines()]
    pmcs = [el for el in pmcs if len(el)]

    params = {'ids': ','.join(pmcs), 'format': 'csv'}
    csv_content = requests.get(url = url, params = params)

    df = pd.read_csv(StringIO(csv_content.text))

    #Fix the entry for PMC4392563 which is not correctly identified
    df.loc[df.PMCID == "PMC4392563", ['PMID', 'DOI']] = '25697273', '10.1007/s11751-015-0211-9'
    
    #Obtain now Semantic Scholar ids
    url = 'https://api.semanticscholar.org/v1/paper/PMID:XXXXX'
    S2ids = []
    for el in df.PMID.apply(str).values.tolist():
        response = requests.get(url = url.replace('XXXXX', el))
        paperdata = json.load(StringIO(response.text))
        S2ids.append(paperdata['paperId'])

    df['S2ID'] = S2ids
    #Save dataframe
    df[['PMCID', 'PMID', 'S2ID', 'DOI']].to_csv(csv_file, index=False) 

    return

def get_pdf_files(PMClist_file, pdf_folder):
    """
    Download PDF files for all PMCs in the file PMClist_file
    :param PMClist_file: Path to text file with full list of PMCs for dataset
    :param pdf_folder: Path to folder where the pdf files will be saved
    """
    
    url = 'https://europepmc.org/backend/ptpmcrender.fcgi?accid=PMCXXXXX&blobtype=pdf'
    
    if not pdf_folder.exists():
        pdf_folder.mkdir()

    with PMClist_file.open() as fin:
        pmcs = [el.strip() for el in fin.readlines()]
    pmcs = [el for el in pmcs if len(el)]
    
    for el in pmcs:
        path2file = pdf_folder.joinpath(el+'.pdf')
        response = requests.get(url = url.replace('PMCXXXXX', el))
        
        if response.ok:
            with path2file.open('wb') as f:
                f.write(response.content)
            printgr('Correctly processed ' + el)
        else:
            printred('Could not retrieve ' + el)

    return

def get_xml_files(PMClist_file, xml_folder):
    """
    Download fulltext XML files for all PMCs in the file PMClist_file
    :param PMClist_file: Path to text file with full list of PMCs for dataset
    :param xml_folder: Path to folder where the fulltext XML files will be saved
    """

    url = 'https://www.ebi.ac.uk/europepmc/webservices/rest/PMCXXXXX/fullTextXML'

    if not xml_folder.exists():
        xml_folder.mkdir()

    with PMClist_file.open() as fin:
        pmcs = [el.strip() for el in fin.readlines()]
    pmcs = [el for el in pmcs if len(el)]

    for el in pmcs:
        path2file = xml_folder.joinpath(el+'.xml')
        response = requests.get(url = url.replace('PMCXXXXX', el))

        if response.ok:
            with path2file.open('w') as f:
                f.write(response.text)
            printgr('Correctly processed ' + el)
        else:
            printred('Could not retrieve ' + el)

    return

def get_annotations(PMClist_file, annotations_folder):
    """
    Download available EuropePMC annotations for all PMCs in the file PMClist_file
    :param PMClist_file: Path to text file with full list of PMCs for dataset
    :param annotations_folder: Path to folder where the annotation files will be saved
    """

    url = 'https://www.ebi.ac.uk/europepmc/annotations_api/annotationsByArticleIds?articleIds=PMC%3AXXXXX&format=JSON'

    if not annotations_folder.exists():
        annotations_folder.mkdir()

    with PMClist_file.open() as fin:
        pmcs = [el.strip() for el in fin.readlines()]
    pmcs = [el for el in pmcs if len(el)]

    for el in pmcs:
        path2file = annotations_folder.joinpath(el+'.json')
        response = requests.get(url = url.replace('XXXXX', el.split('PMC')[1]))

        if response.ok:
            with path2file.open('w') as f:
                f.write(response.text)
            printgr('Correctly processed ' + el)
        else:
            printred('Could not retrieve ' + el)

    return

def get_S2_data(csv_file, S2_folder):
    """
    Download Semantic Scholar description of all papers and authors in base dataset
    :param csv_file: Path to csv file with PMID to PMC translations
    :param Sauthors_folder: Path to folder where the downloaded papers will be saved
    """

    """paperUrl = 'https://api.semanticscholar.org/v1/paper/XXXXX'
    authorUrl = 'https://api.semanticscholar.org/v1/author/XXXXX'
    
    if not S2_folder.exists():
        S2_folder.mkdir()

    df = pd.read_csv(csv_file)

    for idx,el in enumerate(df.S2ID.values.tolist()):
        printgr('Processing paper ' + el + ' (' + str(idx) + ')')
        time.sleep(5)
        response = requests.get(url = paperUrl.replace('XXXXX', el))
        while not response.ok:
            print('Sleep')
            time.sleep(10)
            response = requests.get(url = paperUrl.replace('XXXXX', el))
        paperdata = json.load(StringIO(response.text))
        with S2_folder.joinpath(el +'.json').open('w') as fout:
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
    """
    author_files = sorted([d for d in S2_folder.iterdir() if d.name.startswith('Author') and d.name.endswith('.json')])
    print('Número de autores en el dataset base:', len(author_files))
    author_counts = []
    for af in author_files:
        with af.open() as fin:
            author_data = json.load(fin)
        author_counts.append(len(author_data['papers']))
    print('Número total de artículos asociados a los autores:', sum(author_counts))


    paper_files = sorted([d for d in S2_folder.iterdir() if (not d.name.startswith('Author')) and d.name.endswith('.json')])
    print('Número de papers en el dataset base:', len(paper_files))
    reference_counts = []
    for pf in paper_files:
        with pf.open() as fin:
            paper_data = json.load(fin)
        reference_counts.append(len(paper_data['references']))
    print('Número total de referencias asociadas al dataset base:', sum(reference_counts))



    

    return



if __name__ == "__main__":
    
    Path2data = Path('./data/Agriculture')
    PMClist_file = Path2data.joinpath('PMC_list.txt')
    csv_file = Path2data.joinpath('PMC_PMID_S2.csv')
    pdf_folder = Path2data.joinpath('pdf')
    xml_folder = Path2data.joinpath('xml_fulltext')
    annotations_folder = Path2data.joinpath('EuropePMC_annotations')
    S2_folder = Path2data.joinpath('Extended_AGR')

    ######################################################
    # Extract PMID identifiers of the Agriculture data, in order to access descriptions using S2
    ######################################################
    #PMC_to_PMID_S2(PMClist_file=PMClist_file, csv_file=csv_file)

    ######################################################
    # Download PDF files for all articles
    ######################################################
    #get_pdf_files(PMClist_file=PMClist_file, pdf_folder=pdf_folder)
    
    ######################################################
    # Download XML full text for all articles
    ######################################################
    #get_xml_files(PMClist_file=PMClist_file, xml_folder=xml_folder)

    ######################################################
    # Download PMC annotations
    ######################################################
    #get_annotations(PMClist_file=PMClist_file, annotations_folder=annotations_folder)

    ######################################################
    # Download papers and authors information from S2
    ######################################################
    get_S2_data(csv_file=csv_file, S2_folder=S2_folder)
