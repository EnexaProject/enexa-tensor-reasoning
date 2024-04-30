from tnreason import knowledge
from tnreason import encoding

kb = knowledge.HybridInferer(
    **encoding.load_from_yaml("./fb_backKb.yaml")
)

kb.create_sampleDf(sampleNum=100).to_csv("./fb_sampleDf.csv")
