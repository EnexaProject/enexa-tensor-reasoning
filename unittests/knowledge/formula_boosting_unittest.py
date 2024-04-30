import unittest

from tnreason import knowledge
from tnreason import encoding

import pandas as pd

backKb = knowledge.load_kb_from_yaml("./assets/fb_backKb.yaml")
architecture = encoding.load_from_yaml("./assets/fb_architecture.yaml")


class FormulaBoostingTest(unittest.TestCase):
    def test_als(self):
        booster = knowledge.FormulaBooster(knowledgeBase=backKb,
                                           specDict=encoding.load_from_yaml("./assets/fb_als_boostSpec.yaml"))
        booster.find_candidate(architectureDict=architecture,
                               sampleDf=pd.read_csv("assets/fb_sampleDf.csv"))

    def test_gibbs(self):
        booster = knowledge.FormulaBooster(knowledgeBase=backKb,
                                           specDict=encoding.load_from_yaml("./assets/fb_gibbs_boostSpec.yaml"))
        booster.find_candidate(architectureDict=architecture,
                               sampleDf=pd.read_csv("assets/fb_sampleDf.csv"))
