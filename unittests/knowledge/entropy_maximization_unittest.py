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
        satisfactionDict = knowledge.EmpiricalDistribution(sampleDf).get_satisfactionDict(expressionsDict)

        entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict, backCores={})
        values = entropyMaximizer.alternating_optimization(sweepNum=10)
        for key in values:
            self.assertGreaterEqual(0.1, abs(values[key][-1] - values[key][-2]))

    def test_backCores(self):
        expressionsDict = {"f1": ["imp","a","b"], "f2": ["imp","a","c"], "f3": ["a"]}
        satisfactionDict = knowledge.EmpiricalDistribution(sampleDf).get_satisfactionDict(expressionsDict)
        entropyMaximizer = knowledge.EntropyMaximizer(expressionsDict, satisfactionDict=satisfactionDict,
                                                      backCores=encoding.create_formulas_cores({
                                                          "fact1" : ["imp","a","b"]
                                                      }))
        values = entropyMaximizer.alternating_optimization(sweepNum=2)

        self.assertEqual(0, values["f1"][0])
        self.assertEqual(0, values["f1"][1])