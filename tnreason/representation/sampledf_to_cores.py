import numpy as np
import time

import tnreason.logic.coordinate_calculus as cc

def identify_relevant_columns(columns,argument):
    relcolumns = []
    for column in columns:
        if column.split("(")[1][:-1] == argument:
            relcolumns.append(column)
    return relcolumns

def sampleDf_to_class_values(sampleDf,
                           individual):
    startTime = time.time()
    dataNum = sampleDf.values.shape[0]

    relColumns = identify_relevant_columns(sampleDf.columns,individual)
    coreValues = np.zeros((dataNum,len(relColumns)))

    for i, row in sampleDf.iterrows():
        for column in relColumns:
            if row[column] == 1:
                coreValues[i,relColumns.index(column)] = 1
    endTime = time.time()
    return coreValues, relColumns, endTime-startTime

def sampleDf_to_relation_values(sampleDf,
                              individual1,
                              individual2):
    startTime = time.time()
    dataNum = sampleDf.values.shape[0]

    relColumns = identify_relevant_columns(sampleDf.columns, individual1+","+individual2)
    coreValues = np.zeros((dataNum, len(relColumns), dataNum))

    for i, row in sampleDf.iterrows():
        for column in relColumns:
            if row[column] == 1:
                coreValues[i, relColumns.index(column), i] = 1
    endTime = time.time()
    return coreValues, relColumns, endTime - startTime

def sampleDf_to_universal_core(sampleDf,candidates):
    return sampleDf[candidates].values


def create_fixedCore(sampleDf, candidates, coreColors, coreName):
    return cc.CoordinateCore(sampleDf_to_universal_core(sampleDf, candidates), coreColors, coreName)