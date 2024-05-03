import pandas as pd

sampleDf = pd.read_csv("./bbb_generated.csv")

print(sampleDf.columns)

from tnreason import encoding

print(encoding.create_data_cores(sampleDf,["Moebel(z)","szszur"]))

