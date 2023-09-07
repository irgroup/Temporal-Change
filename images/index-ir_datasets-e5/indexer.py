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


def gen_docs(data, batch_size, field):
    """Generate batches of documents from the WT collection. Creats a global dict of ids to doc ids."""
    global c
    c = 0
    global ids
    ids = {}
    batch = []

    for item in data:
        if len(batch) == batch_size:
            full_batch = batch
            batch = []
            batch.append(item.default_text())
            yield full_batch
        else:
            batch.append(item.default_text())

        ids[c] = getattr(item, f"{field}_id")
        c += 1
    if batch:
        yield batch


def encode(data, index_path, batch_size, num_docs, save_every, mode, stop_at=0):
    """create embeddings for docs in batches and save in batches"""
    print("Start encoding...")

    def save_embs(embs, c):
        embs = torch.cat(embs)
        c_ = "0" + str(c) if len(str(c)) == 1 else str(c)
        torch.save(embs, f"{index_path}/e5_embeddings_{c_}.pt")
        print(f"Saved embeddings for {(c+1)*len(embs)} documents")

    def save_ids(ids):
        with open(index_path + "/ids.json", "w") as f:
            json.dump(ids, f)

    c = 0
    embs = []
    for batch in tqdm(
        data,
        total=(int(num_docs / batch_size)),
    ):
        print(len(batch))
        embeddings = calc_embeddings(batch, mode)
        embs.append(embeddings)

        if len(embs) >= save_every:
            save_embs(embs, c)
            c += 1
            embs = []
        if stop_at:
            if c == stop_at:
                break
    if embs:
        save_embs(embs, c)
    save_ids(ids)
    print(f"Done with encoding")


# load index
def create_index(index_dir, size=768):
    """create index from embedding parts"""
    print("Creating index...")
    files = os.listdir(index_dir)
    files.sort()  # TODO sorting fails, to leading 0

    index = faiss.IndexFlatL2(size)  # build the index

    for file in files:
        if file.endswith(".pt"):
            index.add(torch.load(index_dir + "/" + file).numpy())
    faiss.write_index(index, index_dir + "/index")


def index(
    dataset, index_document_path, index_query_path, model_name, batch_size, save
):
    load_model(model_name)
    dataset = ir_datasets.load(dataset)

    encode(
        data=gen_docs(dataset.docs_iter(), batch_size=batch_size, field="doc"),
        index_path=index_document_path,
        batch_size=batch_size,
        num_docs=dataset.docs_count(),
        mode="passage",
        save_every=save,
    )

    encode(
        data=gen_docs(dataset.queries_iter(), batch_size=batch_size, field="query"),
        index_path=index_query_path,
        batch_size=batch_size,
        num_docs=dataset.queries_count(),
        save_every=save,
        mode="query",
    )

    print("Done with encodng, start indexing...")
    create_index(index_document_path)
    print("Done with indexing")
