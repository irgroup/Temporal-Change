from argparse import ArgumentParser
from indexer import index
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


def setup_index_dir(dataset_name):
    index_path = get_artefact_path(dataset_name)
    os.makedirs(f"{index_path}/documents")
    os.makedirs(f"{index_path}/queries")


def main():
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--dataset_name", help="Name or path to the dataset to be processed", required=True
    )
    args = parser.parse_args()

    setup_index_dir(args.dataset_name)

    index_document_path = get_artefact_path(args.dataset_name) + "/documents"
    index_query_path = get_artefact_path(args.dataset_name) + "/queries"

    index(args.dataset_name, index_document_path, index_query_path)

if __name__ == "__main__":
    main()
