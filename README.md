# Temporal Reproducibility of Retrieval Effectiveness
This repository contains the code to investigate the temporal reproducibility of retrieval effectiveness.
Five state-of-the-art retrieval systems are revisited in different search scenarios with temporal changes in experimental components (such as documents, topics, and qrels), and it is investigated how they reproduce their effectiveness at a later point in time.

This repository holds the code to create and augment the datasets to simulate the different EE, create and run the retrieval systems in Docker containers to produce the runs, and analyze the runs to investigate the temporal reproducibility of retrieval effectiveness. In the following, the usage instructions are listed to reproduce the results.


## Install dependencies

All dependencies are listed in the [requirements.txt](https://github.com/irgroup/Temporal-Persistence/blob/main/requirements.txt) file. To install the dependencies, run the following command:

```bash
pip install -r requirements.txt
```




## 1. Prepare Test Collections
Three test collections are used in the experimental evaluation: TREC-COVID, TripClick, and LongEval. TREC-COVID and TripClick are acquired through [IR Datasets](https://ir-datasets.com) and are expected to be placed in the `data` directory (the expected directory structure is listed later). 
The LongEval dataset is available through LINDAT ([Train](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5010), [Test](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5139)). For further information, we refer to the [LongEval website](https://clef-longeval.github.io/data/). The dataset should also be placed in the `data` directory. 

### 1.1 Scrape metadata for TripClick
Since TripClick does not contain natural rounds, the evolving test collection is simulated by splitting the collection into three parts based on the publication date of the documents. The publication date can be obtained through the [scrape_trip_click_metadata.ipynb](https://github.com/irgroup/Temporal-Persistence/tree/main/scripts/scrape_trip_click_metadata.ipynb) notebook.

### 1.2 Create EEs
To prepare the EEs from the datasets some preprocessing steps are necessary. In the following notebook, the topics are limited to the topics that are used in all EEs and seperate qrels files are created for each EE. The notebook can be found here:
[prepare_datasets.ipynb](https://github.com/irgroup/Temporal-Persistence/tree/main/scripts/prepare_datasets.ipynb)



## 2. Create Runs
The runs are created in a two-step procedure. First, the index is created, and second, the runs are made. These steps can conveniently be executed through docker compose files that are available in the [images directory](https://github.com/irgroup/Temporal-Persistence/tree/main/images)

### 2.1 Create Indexes
The indexes are created through the two files [make-indexe-pyterrier.yml](https://github.com/irgroup/Temporal-Persistence/tree/main/images/make-indexe-pyterrier.yml) and [make-indexe-pyterrier-d2q.yml](https://github.com/irgroup/Temporal-Persistence/tree/main/images/make-indexe-pyterrier-d2q.yml). The compose files will build the containers for the different EEs and execute the indexing. The `data` dir is mounted to the container to access the datasets and to store indexes.
The compose files can be executed with the following command:

```bash
docker compose -f make-indexe-pyterrier.yml up && docker compose -f make-indexe-pyterrier-d2q.yml up
```


### 2.2 Query the Systems
After the indexes are created, the runs are made through the other compose files in the [images directory](https://github.com/irgroup/Temporal-Persistence/tree/main/images). The compose files will build the containers for the different EEs and execute the retrieval. The `data` dir is mounted to the container to access the datasets and indexes and finally to store the runs.

```bash
docker compose -f make-runs-bm25.yml up && \
docker compose -f make-runs-bm25_colbert.yml up && \
docker compose -f make-runs-bm25_monot5.yml up && \
docker compose -f make-runs-bm25_bo1.yml up && \
docker compose -f make-runs-pl2.yml up && \
docker compose -f make-runs-xsqra_m.yml up && \
docker compose -f make-runs-rrf.yml up
```


## 3. Experimental Evaluation
After the runs are created, it can be investigated how well they reproduce the retrieval effectiveness at a later point in time.

### 3.1 Filter runs and qrels
In the experimental evaluation, the scenario of changing documents (D'TQ') is investigated. To achieve this, the datasets need to be limited to the set of queries that are used in all EEs. This can be achieved through the following notebook: [prepare_runs.ipynb](https://github.com/irgroup/Temporal-Persistence/tree/main/scripts/prepare_runs.ipynb).



### 3.2 Analyze the runs
Finally, the code for the analysis of the runs can be found in the following notebooks:
- [Prepare dataset statistics](https://github.com/irgroup/Temporal-Persistence/tree/main/scripts/prepare_dataset_statistics.ipynb)
- [Analyze the test collections EEs](https://github.com/irgroup/Temporal-Persistence/tree/main/evaluation/analyze_datasets.ipynb)
- [Analyze the reproduction of retrieval effectiveness](https://github.com/irgroup/Temporal-Persistence/tree/main/evaluation/analyze_runs.ipynb)



## Expected Data Structure
```
data
├── core_queries.tsv
├── dataset
│   └── TripClick
│       ├── ...
│       ...
│       
├── index
│   ├── index-longeval-LT-pyterrier
│   ...    
│   
├── longeval_topics_core_queries_unified.tsv
├── metadata.jsonl
├── metadata_processed.jsonl
│   
├── qrels
│   ├── longeval-LT.qrels-test
│   ...
│   
├── run-core_queries
│   ├── run-longeval-LT-test-bm25-pyterrier
│   ...
│   
├── trec-covid.core-queries.txt
├── trec_covid.json
└── tripclick-subcollections.json
```