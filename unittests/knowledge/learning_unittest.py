import unittest

from tnreason import knowledge

genKB = knowledge.HybridKnowledgeBase(
    facts={"f1": ["a1"]},
    weightedFormulas={
        "wf1": ["imp", "a1", "a2", 1.1424],
        "wf1.5": ["a2", 0.2],
        "wf2": ["not", "a3", 50.2]
    }
)
sampleDf = knowledge.InferenceProvider(genKB).draw_samples(100)


class HybridLearnerTest(unittest.TestCase):
    def test_boosting(self):
        learner = knowledge.HybridLearner(knowledge.HybridKnowledgeBase(
            weightedFormulas={"w1": ["not", "a3", 2],
                              "w2": ["a2", -1]}
        ))
        learner.graft_formula({
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
        }, sampleDf, stepName="_funBoost")
        hybridKB = learner.get_knowledge_base()

        self.assertEqual(hybridKB.weightedFormulas["neur1_funBoost"][:-1], ["imp", "a1", "a2"])
        self.assertEqual(hybridKB.facts["w1"], ["not", "a3"])
