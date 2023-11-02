import pandas as pd

def sampleDf_to_pairDf(sampleDf,prefix):
    columns = sampleDf.columns
    dataNum = len(sampleDf)
    rootClasses = find_rootClasses(columns)
    pairDf = pd.DataFrame(data= {entry:[prefix+str(entry)+"_"+str(i) for i in range(dataNum)] for entry in rootClasses},
                          columns=rootClasses)
    return pairDf

def find_rootClasses(columns):
    rootClasses = []
    for column in columns:
        argpart = column.split("(")[1][:-1]
        if "," in column:
            arg1, arg2 = argpart.split(",")
            if arg1 not in rootClasses:
                rootClasses.append(arg1)
            if arg2 not in rootClasses:
                rootClasses.append(arg2)
        else:
            if argpart not in rootClasses:
                rootClasses.append(argpart)
    return rootClasses