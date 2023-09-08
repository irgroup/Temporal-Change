from argparse import ArgumentParser
from system import rank
import os

SYSTEM = os.getenv("SYSTEM")
METHOD = os.getenv("METHOD")

def run_system(run_name):
    return "-".join(run_name.split("-")[1:3])


def run_path(index):
    system = "rrf("
    for path in index:
        system += run_system(path) + "-"
    system += ")"
    return f"/data/run/run-{system}-{METHOD}"


def main():
    parser = ArgumentParser(description="")
    parser.add_argument(
        "--index", help="Name or path to the dataset to be processed", required=True, nargs='+')
    parser.add_argument(
        "--queries", help="Name of the query file", required=False, default="queries")
    
    args = parser.parse_args()
    print(args.index)
    result_path = run_path(args.index)

    rank(args.index, args.queries, result_path)

if __name__ == "__main__":
    main()
