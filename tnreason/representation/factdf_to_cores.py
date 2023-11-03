import numpy as np
import time

import tnreason.representation.ttl_to_csv as ttodf


def generate_factDf(ttlPath, limit=None):
    return ttodf.generate_csv_list(ttlPath, limit)


def filter_df(df, column, values):
    startTime = time.time()
    df = df[df[column].isin(values)]
    endTime = time.time()
    return df, endTime - startTime

def prefixify(lis, prefix):
    return np.array([prefix+entry.split("(")[0] for entry in lis])


def factDf_to_class_values(df, individuals=None, classes=None, prefix=""):
    startTime = time.time()

    if classes is not None:
        classes = prefixify(classes,prefix)

    classes = np.array(classes)
    individuals = np.array(individuals)

    classDf = df[df["predicate"].isin(["http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "rdf:type"])][
        ["subject", "object"]]

    if classes is None:
        classes = np.unique(df["object"].values)
    if individuals is not None:
        classDf, latency = filter_df(classDf, "subject", individuals)
    else:
        individuals = np.unique(classDf["subject"].values)

    repTensor = np.zeros((len(individuals), len(classes)))
    for i, row in classDf.iterrows():
        subPos = np.argwhere(individuals == row["subject"])
        obPos = np.argwhere(classes == row["object"])
        repTensor[subPos, obPos] = 1
    endTime = time.time()

    return repTensor, endTime - startTime


def factDf_to_relation_values(df, individuals1=None, individuals2=None, relations=None, prefix=""):
    startTime = time.time()

    if prefix != "":
        if relations is not None:
            relations = prefixify(relations,prefix)

    if individuals1 is not None:
        df, latency = filter_df(df, "subject", individuals1)
    else:
        individuals1 = np.unique(df["subject"].values)

    if individuals2 is not None:
        df, latency = filter_df(df, "object", individuals2)
    else:
        individuals2 = np.unique(df["object"].values)

    if relations is not None:
        df, latency = filter_df(df, "predicate", relations)
    else:
        relations = np.unique(df["predicate"].values)

    relations = np.array(relations)
    individuals1 = np.array(individuals1)
    individuals2 = np.array(individuals2)

    repTensor = np.zeros((len(individuals1), len(relations), len(individuals2)))
    for i, row in df.iterrows():
        subPos = np.argwhere(individuals1 == row["subject"])
        predPos = np.argwhere(relations == row["predicate"])
        obPos = np.argwhere(individuals2 == row["object"])
        repTensor[subPos, predPos, obPos] = 1
    endTime = time.time()

    return repTensor, endTime - startTime
