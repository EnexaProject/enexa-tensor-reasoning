from representations import factdf_to_cores as ftoc
from representations import sampledf_to_cores as stoc

import pandas as pd

## stoc tests

sampleDf = pd.read_csv("./tests/assets/bbb_generated.csv", index_col=0)

relColumns = stoc.identify_relevant_columns(sampleDf.columns,"z")
assert relColumns == ['Unterschrank(z)', 'Moebel(z)'], "SampleDf Column Selection is Wrong!"

classValues, relColumns, latency = stoc.sampleDf_to_class_values(sampleDf,"z")
assert classValues.shape == (100,2), "SampleDf Class Core Values have wrong shape {}!".format(classValues.shape)

relationValues, relColumns, latency = stoc.sampleDf_to_relation_values(sampleDf,"x","y")
assert relationValues.shape == (100,1,100), "SampleDf Relation Core Values have wrong shape {}!".format(relationValues.shape)

## ftoc test
factDf = ftoc.generate_factDf("./tests/assets/bbb_generated.ttl")
classValues, latency = ftoc.factDf_to_class_values(factDf)
assert classValues.shape == (254, 195), "FactDf Class Core Values have wrong shape {}!".format(classValues.shape)

relationValues, latency = ftoc.factDf_to_relation_values(factDf)
assert relationValues.shape == (323, 5, 195), "FactDf Relation Core Values have wrong shape {}!".format(relationValues.shape)