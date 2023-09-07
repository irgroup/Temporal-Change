import json
import os

import faiss
import ir_datasets
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def fix_ir_dataset_naming(dataset):
    return "-".join(dataset.split("/")[-2:])


def load_model(model_name):
    global tokenizer, model, device
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    # prepare model for gpu use
    print("GPU available: ", torch.cuda.is_available())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = model.to(device)


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


@torch.no_grad()
def calc_embeddings(texts, mode="passage"):
    input_texts = [f"{mode}: {text}" for text in texts]
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )
    for key, val in batch_dict.items():
        batch_dict[key] = batch_dict[key].to(device, non_blocking=True)

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    return embeddings.detach().cpu()  # .numpy()


def gen_docs(doc_path, batch_size):
    global c
    c = 0
    global ids
    ids = {}
    batch = []
    for filename in os.listdir(doc_path):
        with open(doc_path + "/" + filename, "r") as f:
            for line in f:
                l = json.loads(line)
                for doc in l:
                    if len(batch) == batch_size:
                        full_batch = batch
                        batch = []
                        batch.append(doc["contents"])
                        yield full_batch
                    else:
                        batch.append(doc["contents"])
                    ids[c] = doc["id"]
                    c += 1
    if batch:
        yield batch


def encode_queries(queries_path, index_query_path, dataset_slice, batch_size):
    queries = pd.read_csv(queries_path, sep="\t", header=None)
    ids = {}
    batch = []
    embs = []
    for id, row in queries.iterrows():
        ids[id] = row[0]
        query = row[1]

        batch.append(query)
        if len(batch) == batch_size:
            query_embedding = calc_embeddings(batch, mode="query")
            embs.append(query_embedding)
            batch = []
    if batch:
        query_embedding = calc_embeddings(batch, mode="query")
        embs.append(query_embedding)

    # save embs
    embs = torch.cat(embs)
    assert len(embs) == len(ids), "Queries: embedding and ids len offsett" 
    torch.save(embs, f"{index_query_path}/e5_embeddings-{dataset_slice}.pt")
    with open(f"{index_query_path}/ids.json", "w") as f:
        json.dump(ids, f)

    print(f"Done with encoding {dataset_slice} queries")


def encode(data, index_path, batch_size, num_docs, save_every, stop_at=3):
    def save_embs(embs, c, batch_size):
        embs = torch.cat(embs)
        torch.save(embs, f"{index_path}/e5_embeddings_{c}.pt")
        print(f"Saved embeddings for {c*batch_size} documents")

    def save_ids(ids):
        with open(os.path.join(index_path, "ids.json"), "w") as f:
            json.dump(ids, f)

    c = 0
    embs = []
    for batch in tqdm(data, total=(int(num_docs / batch_size))):
        embeddings = calc_embeddings(batch)
        embs.append(embeddings)

        if len(embs) >= save_every:
            save_embs(embs, c, batch_size)
            c += 1
            embs = []
        if stop_at:
            if c == stop_at:
                break
    if embs:
        save_embs(embs, c, save_every)
    save_ids(ids)
    print(f"Done with encoding documents")


# load index
def create_index(index_dir, size=768):
    files = os.listdir(index_dir)
    files.sort()

    index = faiss.IndexFlatL2(size)

    for file in files:
        if file.endswith(".pt"):
            index.add(torch.load(index_dir + "/" + file).numpy())
    faiss.write_index(index, index_dir + "/index")
    print("Done with indexing")


def get_dataset_subcollection(dataset):
    return dataset.split("-")[1]


def get_documents_path(dataset):
    subcollection = get_dataset_subcollection(dataset)
    if subcollection == "LT":
        return "/data/dataset/LongEval/test-collection/B-Long-September/English/Documents/Json/"
    elif subcollection == "ST":
        return "/data/dataset/LongEval/test-collection/B-Long-September/English/Documents/Json/"
    elif subcollection == "WT":
        return "/data/dataset/LongEval/publish/English/Documents/Json/"


def get_querie_path(dataset):
    subcollection = get_dataset_subcollection(dataset)
    if subcollection == "LT":
        return {
            "test": "/data/dataset/LongEval/test-collection/B-Long-September/English/Queries/test09.tsv"
        }
    elif subcollection == "ST":
        return {
            "test": "/data/dataset/LongEval/test-collection/A-Short-July/English/Queries/test07.tsv"
        }
    elif subcollection == "WT":
        return {
            "train": "/data/dataset/LongEval/publish/English/Queries/train.tsv",
            "test": "/data/dataset/LongEval/publish/English/Queries/heldout.tsv",
        }


def index(
    dataset, index_document_path, index_query_path, model_name, batch_size, save
):
    load_model(model_name)

    data = gen_docs(doc_path=get_documents_path(dataset), batch_size=batch_size)
    encode(
        data=data,
        index_path=index_document_path,
        batch_size=batch_size,
        num_docs=1570734,
        save_every=save,
    )

    queries_paths = get_querie_path(dataset)
    for dataset_slice, path in queries_paths.items():
        encode_queries(path, index_query_path, dataset_slice, batch_size)

    create_index(index_document_path)
