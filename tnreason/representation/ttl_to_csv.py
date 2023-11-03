import rdflib
import pandas as pd
import time

def generate_csv_list(ttlPath, limit=None):
    return extend_csv_list(pd.DataFrame(columns=["subject", "predicate", "object"]), ttlPath, limit=limit)


def extend_csv_list(factDf, ttlPath, limit=None):
    g = rdflib.Graph()
    g.parse(ttlPath)

    print("Loading {} successfull.".format(ttlPath))
    print("Graph has {} facts.".format(len(g)))

    reftime = time.time()
    factCount = 0
    for fact in g:
        if time.time() - reftime > 30:
            print("Loaded {} facts.".format(factCount))
            reftime = time.time()
        if limit is not None:
            if factCount > limit:
                return factDf
        rowDf = pd.DataFrame({
            "subject": [str(fact[0])],
            "predicate": [str(fact[1])],
            "object": [str(fact[2])]
            }, index = [factCount])
        factDf = pd.concat([factDf,rowDf])
        factCount += 1
    return factDf


if __name__ == "__main__":

    starttime = time.time()
    df = generate_csv_list("./results/knowledge_graph/parquet_tev2.ttl", limit=50000)
    endtime = time.time()
    print("Generation done after {} seconds.".format(endtime - starttime))

    df.to_csv("./results/csv_files/parquet_tev2_list.csv")
    savetime = time.time()
    print("Saving done after additional {} seconds.".format(savetime - endtime))
