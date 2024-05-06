import unittest

from tnreason import encoding
from tnreason import knowledge

generatingKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(weightedFormulas=
{
    "f1": ["imp","a","b", 2.567],
    "f2": ["imp","a","c", 2.222],
    "f3": ["a", 1.78]
}))

sampleNum = 200
sampleDf = generatingKB.draw_samples(sampleNum)


class EntropyMaximationTest(unittest.TestCase):
    def test_convergence(self):
        expressionsDict = {"f1": ["imp","a","b"], "f2": ["imp","a","c"], "f3": ["a"]}
        inferer = knowledge.InferenceProvider(knowledge.EmpiricalDistribution(sampleDf))
        satisfactionDict = {key: inferer.ask(expressionsDict[key]) for key in expressionsDict}

        entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict, backCores={})
        weights, facts = entropyMaximizer.alternating_optimization(sweepNum=10)
        for key in weights:
            self.assertGreaterEqual(0.1, abs(weights[key][-1] - weights[key][-2]))

    def test_backCores(self):
        expressionsDict = {"f1": ["imp","a","b"], "f2": ["imp","a","c"], "f3": ["a"]}
        inferer = knowledge.InferenceProvider(knowledge.EmpiricalDistribution(sampleDf))
        satisfactionDict = {key: inferer.ask(expressionsDict[key]) for key in expressionsDict}

        entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict,
                                                      backCores=encoding.create_formulas_cores({
                                                          "fact1" : ["imp","a","b"]
                                                      }))
        weights, facts = entropyMaximizer.alternating_optimization(sweepNum=2)

        self.assertEqual(0, weights["f1"][0])
        self.assertEqual(0, weights["f1"][1])