# Temporal Reproducibility of Retrieval Effectiveness


## 1. Prepare Test Collections
Since in the experimental evaluation, the scenario D'TQ' is investigated, the datasets need to be prepared before the runs can be created:

- [Scrape metadata for TripClick](https://github.com/irgroup/Temporal-Persistence/tree/main/scripts/scrape_trip_click_metadata.ipynb)

- [Prepare EEs](https://github.com/irgroup/Temporal-Persistence/tree/main/scripts/create_qrels.ipynb)


## 2. Create Runs
The runs are created through a Docker container in a two-step procedure. First, the index is created, and second, the runs are made. In the [images directory](https://github.com/irgroup/Temporal-Persistence/tree/main/images), besides the docker files, two docker-compose files can be found to create the PyTerrier indexes [with d2q](https://github.com/irgroup/Temporal-Persistence/tree/main/images/make-indexe-pyterrier-d2q.yml) and [without](https://github.com/irgroup/Temporal-Persistence/tree/main/images/make-indexe-pyterrier.yml). Further different compose files are available to create the runs.



## 3. Experimental Evaluation
After the runs are created, it can be investigated how well they reproduce the retrieval effectiveness at a later point in time. This is done through the notebooks:
- [Filter runs](https://github.com/irgroup/Temporal-Persistence/tree/main/scripts/filter_data.ipynb)

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