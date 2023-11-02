import pandas as pd
import numpy as np
import time

def pairDf_to_target_values(pairDf, individualsdict, individuals):
    startTime = time.time()

    for individual in individuals:
        if individualsdict[individual] is not None:
            pairDf = pairDf[pairDf[individual].isin(individualsdict[individual])]
        else:
            individualsdict[individual] = pairDf[individual].values
        individualsdict[individual] = np.array(individualsdict[individual])

    repTensor = np.zeros(shape=[len(individualsdict[individual]) for individual in individuals])
    for i, row in pairDf.iterrows():
        position = tuple(int(np.argwhere(individualsdict[individual]==row[individual])) for individual in individuals)
        repTensor[position] = 1
    endTime = time.time()
    return repTensor, endTime - startTime