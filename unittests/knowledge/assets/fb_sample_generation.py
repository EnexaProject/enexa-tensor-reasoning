from tnreason import knowledge

kb = knowledge.InferenceProvider(knowledge.load_kb_from_yaml("./fb_backKb.yaml"))
kb.draw_samples(sampleNum=100).to_csv("./fb_sampleDf.csv")
