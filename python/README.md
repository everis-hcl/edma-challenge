


# Python scritps

## 1. Prerequisites
 * [Python 3.8](https://www.python.org/downloads/)
 * [Anaconda](https://docs.anaconda.com/anaconda/install/)
 * [virtualenv](https://pypi.org/project/virtualenv/)

## 2. Scripts

The tool includes three executable python scripts:

* **`ml_labeler`**: ML labeler source code and  guidelines.
* **`CreateDatasetAGR.py`**: Generate the dataset for AGR corpus.
* **`CreateDatasetBIO.py`**: Generate the dataset for BIO corpus.
* **`computeTFIDF_AGR.py`**: Calculate TFIDF for AGR corpus.
*  **`computeTFIDF_BIO.py`**: Calculate TFIDF for BIO corpus.
*  **`computeTFIDF_GIT.py`**: Calculate TFIDF for GIT corpus.
* **`TrainModelsAGR.py`**: Train models for AGR corpus.
*  **`TrainModelsBIO.py`**: Train models for BIO corpus.



## 3. Requirements

To run the scripts, you should:
* **`Create an virtual environment`**: py -m venv env
* **`Activaite the venv`**: ./env/Scripts/activate
* **`Install the dependencies`**: pip install -r requirements.txt
* **`Install the dbManager`**: from ./dbManager or from [here](https://github.com/jeroarenas/dbManager).
* **`Install the Mallet`**: [http://mallet.cs.umass.edu/](http://mallet.cs.umass.edu/)

Copy in the data_file folder in the python directory (4 directories are added `corpus`, `data`, `models` and `TFIDFcorpus`).

## 3. Run the scripts:

Once you have installed everything, you can run the script in the following order:
* **`CreateDatasetAGR.py`**: Generate the dataset for AGR corpus.
* **`CreateDatasetBIO.py`**: Generate the dataset for BIO corpus.
* **`computeTFIDF_AGR.py`**: Calculate TFIDF for AGR corpus.
*  **`computeTFIDF_BIO.py`**: Calculate TFIDF for BIO corpus.
*  **`computeTFIDF_GIT.py`**: Calculate TFIDF for GIT corpus.
* **`TrainModelsAGR.py`**: Train models for AGR corpus.
*  **`TrainModelsBIO.py`**: Train models for BIO corpus.
* **`ml_labeler`**: ML labeler source code and  guidelines.
