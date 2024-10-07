import unittest

from tnreason import knowledge
from tnreason import encoding

import pandas as pd

backKb = knowledge.load_kb_from_yaml("./assets/fb_backKb.yaml")
architecture = encoding.load_from_yaml("./assets/fb_architecture.yaml")

genKB = knowledge.HybridKnowledgeBase(
    facts={"f1": ["a1"]},
    weightedFormulas={
        "wf1": ["imp", "a1", "a2", 1.1424],
        "wf1.5": ["a2", 0.2],
        "wf2": ["not", "a3", 5.2]
    }
)
sampleDf = knowledge.InferenceProvider(genKB).draw_samples(100)


class FormulaBoostingTest(unittest.TestCase):
    def test_yaml_als(self):
        booster = knowledge.FormulaBooster(knowledgeBase=backKb,
                                           specDict={**encoding.load_from_yaml("./assets/fb_als_boostSpec.yaml"),
                                                     "headNeurons": ["neur1"], "architecture": architecture})
        booster.find_candidate(sampleDf=pd.read_csv("assets/fb_sampleDf.csv"))

    def test_yaml_gibbs(self):
        booster = knowledge.FormulaBooster(knowledgeBase=backKb,
                                           specDict={**encoding.load_from_yaml("./assets/fb_gibbs_boostSpec.yaml"),
                                                     "headNeurons": ["neur1"], "architecture": architecture})
        booster.find_candidate(sampleDf=pd.read_csv("assets/fb_sampleDf.csv"))

    def test_exact_implication_finding(self):
        booster = knowledge.FormulaBooster(knowledgeBase=knowledge.HybridKnowledgeBase(),
                                           specDict= {
            "method": "exactEnergyMax",
            "sweeps": 10,
            "headNeurons": ["neur1"],
            "architecture":
                {"neur1": [["imp"],
                           ["a1"],
                           ["a3", "a2"]]
                 },
            "acceptanceCriterion": "always",
            "calibrationSweeps": 2
        })
        booster.find_candidate(sampleDf=sampleDf)
        self.assertEquals(booster.candidates["neur1"][-1],"a2")

    def test_gibbs_implication_finding(self):
        booster = knowledge.FormulaBooster(knowledgeBase=knowledge.HybridKnowledgeBase(),
                                               specDict={
                                                   "method": "gibbsSample",
                                                   "sweeps": 10,
                                                   "headNeurons": ["neur1"],
                                                   "architecture":
                                                       {"neur1": [["imp"],
                                                                  ["a1"],
                                                                  ["a3", "a2"]]
                                                        },
                                                   "acceptanceCriterion": "always",
                                                   "calibrationSweeps": 2
                                               })
        booster.find_candidate(sampleDf=sampleDf)
        self.assertEquals(booster.candidates["neur1"][-1], "a2")

