from argparse import ArgumentParser
from indexer import index, fix_ir_dataset_naming
import os

METHOD = os.getenv("METHOD")
TYPE = os.getenv("TYPE")


def get_artefact_path(artefact_name):
    artefact_type = TYPE
    if artefact_type == "index":
        return (f"./data/index/index-{artefact_name}-{METHOD}")
    elif artefact_type == "run":
        return f"./data/runs/"
    elif artefact_type == "model":
        return f"./data/models/model-{artefact_name}-{METHOD}"
    else:
        raise ValueError(f"Artefact type `{artefact_type}` not supported!")


def setup_index_dir(dataset):
    index_path = get_artefact_path(dataset)
    os.makedirs(f"{index_path}/documents")
    os.makedirs(f"{index_path}/queries")


def main():
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--dataset", help="Name or path to the dataset to be processed", required=True
    )
    args = parser.parse_args()

    dataset = fix_ir_dataset_naming(args.dataset)
    setup_index_dir(dataset)

    index_document_path = get_artefact_path(dataset) + "/documents"
    index_query_path = get_artefact_path(dataset) + "/queries"

    index(args.dataset, index_document_path, index_query_path)

if __name__ == "__main__":
    main()
