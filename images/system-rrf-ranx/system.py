from ranx import Run, fuse
import pandas as pd

def rank(index, queries, result):
    runs = []
    for run in index:
        runs.append(Run.from_file(run, kind="trec"))

    run_rrf = fuse(runs=runs, method="rrf")
    
    # run_rrf.name = run_tag
    run_rrf.save(result, kind="trec")

    # limit to 1000 documents per query
    df = pd.read_csv(result, sep=" ", header=None)
    df.sort_values([0,4],ascending=False).groupby(0).head(1000).to_csv(result, sep=" ", header=None, index=False)