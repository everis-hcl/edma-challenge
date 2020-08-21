# Java apps

This directory contains the source file needed to perform keyword mappings against SPARQL endpoints for 
- Agrovoc
- Mesh
- DBPedia
- EuroSciVoc

### Prerequisites

The project needs the following tools:

- Java JDK (8 or higher)
- Apache Maven (3.6 or higher)
- Python (3.8 or higher)

### Build application

In order to build the application locally, you need to go to `java\sparql-mapping` directory. Once there, launch the following command:

```
mvn clean install
```

### Run application

In order to run this java application locally, you need to go to `java\sparql-mapping` directory, open a command prompt or windows shell and execute:
```
python RunSPARQLMapperAGR.py
```
This will run the mapping for all AGR data and produce mapping excel files.
Input Data Folder: data/TFIDFcorpus/AGR
Output Folder: data/ClassificationResults/AGR
Once this script finishes, launch the next script - 

```
python RunSPARQLMapperBIO.py
```
This will run the mapping for all BIO data and produce mapping excel files.
Input Data Folder: data/TFIDFcorpus/BIO
Output Folder: data/ClassificationResults/BIO
Once this script finishes, launch the next script - 

```
python RunSPARQLMapperGIT.py
```This will run the mapping for all GIT data and produce mapping excel files.
Input Data Folder: data/TFIDFcorpus/GIT
Output Folder: data/ClassificationResults/GIT

### Final Results
The final results of the mappings can be seen under:
- AGR - data/FinalResults/AGR
- BIO - data/FinalResults/BIO
- GIT - data/FinalResults/GIT