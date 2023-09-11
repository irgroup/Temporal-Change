import os
import shutil

import pyterrier as pt
import pyterrier_doc2query

if not pt.started():
    pt.init()

METHOD = os.getenv("METHOD")


def get_dataset_subcollection(dataset):
    return dataset.split("-")[1]


def get_querie_path(dataset):
    subcollection = get_dataset_subcollection(dataset)
    if subcollection == "LT":
        return {
            "test": "/data/dataset/LongEval/test-collection/B-Long-September/English/Queries/test09.trec"
        }
    elif subcollection == "ST":
        return {
            "test": "/data/dataset/LongEval/test-collection/A-Short-July/English/Queries/test07.trec"
        }
    elif subcollection == "WT":
        return {
            "train": "/data/dataset/LongEval/publish/English/Queries/train.trec",
            "test": "/data/dataset/LongEval/publish/English/Queries/heldout.trec",
        }


def get_documents_path(dataset):
    subcollection = get_dataset_subcollection(dataset)
    if subcollection == "LT":
        return "/data/dataset/LongEval/test-collection/B-Long-September/English/Documents/Trec/"
    elif subcollection == "ST":
        return "/data/dataset/LongEval/test-collection/A-Short-July/English/Documents/Trec/"
    elif subcollection == "WT":
        return "/data/dataset/LongEval/publish/English/Documents/Trec/"


def move_queries(dataset, index_query_path):
    querie_paths = get_querie_path(dataset)
    for key, path in querie_paths.items():
        shutil.copy(path, f"{index_query_path}/{key}.trec")



def index(dataset, index_document_path, index_query_path, batch_size, num_samples):
    move_queries(dataset, index_query_path)

    documents_path = get_documents_path(dataset)
    documents = [
        os.path.join(documents_path, path) for path in os.listdir(documents_path)
    ]
    gen = pt.index.treccollection2textgen(
        documents,
        num_docs = 1500000,
        verbose=True,
        meta=["docno", "text"],
        tag_text_length= 100000,
        meta_tags={"text": "ELSE"}
        )

    doc2query = pyterrier_doc2query.Doc2Query(
        batch_size=batch_size,
        append=True,
        num_samples=num_samples,
        # verbose=True,
        fast_tokenizer=True,
    )

    indexer = pt.IterDictIndexer(
        index_path=index_document_path,
        verbose=True,
        meta={"docno": 26, "text": 100000},
        meta_tags={"text": "ELSE"},
        )
    
    pipeline = doc2query >> indexer

    indexref = pipeline.index(gen)

    index = pt.IndexFactory.of(indexref)

    print("Indexing done\n_________________________________")
    print(index.getCollectionStatistics().toString())
