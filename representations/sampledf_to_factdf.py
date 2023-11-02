import pandas as pd


def sampleDf_to_factDf(sampleDf,prefix=""):
    columns = sampleDf.columns
    df = pd.DataFrame(columns = ["subject","predicate","object"])
    index = 0
    for i, row in sampleDf.iterrows():
        for column in columns:
           triple = generate_triple(i,column,row[column],prefix = prefix)
           if triple is not None:
                    row_dict = {"subject": triple[0],
                                "predicate": triple[1],
                                "object": triple[2]
                    }
                    row_df = pd.DataFrame(row_dict, index=[index])
                    index +=1
                    df = pd.concat([df, row_df])
    return df






## From createDG:
def generate_triple(index,key,value,prefix):
    if "," in key:
        triple = generate_relation_triple(index,key,value,prefix)
    else:
        triple = generate_membership_triple(index,key,value,prefix)
    return triple

def generate_relation_triple(index,key,value,prefix):
    if value == 1:
        predpart, argpart = key.split("(")
        arg1, arg2 = argpart.split(",")

        s = prefix + arg1 + "_" + str(index)
        p = prefix + predpart
        o = prefix + arg2[:-1] + "_" + str(index)
        return s, p, o
    else:
        return None

def generate_membership_triple(index,key,value,prefix):
    if value == 1:
        predpart, argpart = key.split("(")

        s = prefix + argpart[:-1] + "_" + str(index)
        p = "rdf:type"
        o = prefix + predpart
        return s, p, o
    else:
        return None