import unittest

from tnreason import knowledge

sampleRepetition = 10


class HybridKBTest(unittest.TestCase):
    def test_is_satisfiable(self):
        kb = knowledge.HybridKnowledgeBase(
            weightedFormulas={"e": ["a1", 2]},
            facts={"c1": ["a1"],
                   "c2": ["imp", "a1", "a2"]})
        self.assertTrue(kb.is_satisfiable())

    ## Functionality no longer supported!
    # def test_satisfiability2(self):
    #
    #     with self.assertRaises(ValueError, msg="The initialized Knowledge Base is inconsistent!"):
    #         knowledge.InferenceProvider(facts={"c1": ["a1"], "c2": ["not", "a1"]})

    def test_ask_constraint_entailed(self):
        kb = knowledge.HybridKnowledgeBase(weightedFormulas={"e": ["a1", 2]},
                                           facts={"c1": ["a1"]})
        self.assertEqual("entailed",
                         knowledge.InferenceProvider(kb).ask_constraint("a1")
                         )

    def test_ask_constraint_contradicted(self):
        kb = knowledge.HybridKnowledgeBase(
            weightedFormulas={"e": ["imp", ["eq", "a1", "a2"], ["xor", "a3", "a1"], 2]},
            facts={"c1": ["and", "a1", "a2"]})
        self.assertEqual("contradicting",
                         knowledge.InferenceProvider(kb).ask_constraint(
                             ["not", "a1"])
                         )

    def test_map_query(self):
        kb = knowledge.HybridKnowledgeBase(
            weightedFormulas={"e": ["imp", ["eq", "a1", "a2"], ["xor", "a3", "a1"], 2]},
            facts={"c1": "a1",
                   "c2": ["not", "a2"]})
        self.assertEqual({"a1": 1, "a2": 0},
                         knowledge.InferenceProvider(kb).exact_map_query(["a1", "a2"], evidenceDict={"a3": 1})
                         )

    def test_empty_dicts(self):
        kb = knowledge.HybridKnowledgeBase(
            weightedFormulas={}, facts={})
        self.assertEqual(1,
                         knowledge.InferenceProvider(kb).query(["a1"], evidenceDict={"a1": 1}).values[1])
        self.assertEqual(0.5,
                         knowledge.InferenceProvider(knowledge.HybridKnowledgeBase()).query(["a1"],
                                                                                            evidenceDict={}).values[
                             1])
        self.assertEqual(0.125,
                         knowledge.InferenceProvider(knowledge.HybridKnowledgeBase()).query(["a1", "a3", "a2"],
                                                                                            evidenceDict={}).values[
                             1, 0, 1])

    ## Sampling on facts tests
    def test_not(self):
        hybridKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(
            weightedFormulas={},
            facts={"constraint1": ["not", "a1"]})
        )
        self.assertEqual(0,
                         hybridKB.ask("a1"))

        for rep in range(sampleRepetition):
            sample = hybridKB.forward_sample(["a1"])
            self.assertEqual(0, sample["a1"])

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1"])
            self.assertEqual(0, sample["a1"])

    def test_and(self):
        hybridKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(
            weightedFormulas={"f1": ["a1", 1]},
            facts={"constraint1": ["and", "a1", "a2"]})
        )
        self.assertEqual(0,
                         hybridKB.ask(["not", "a1"]))

        for rep in range(sampleRepetition):
            sample = hybridKB.forward_sample(["a1", "a2"])
            self.assertTrue((int(sample["a1"]) + int(sample["a2"])) == 2)

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2"])
            self.assertTrue((int(sample["a1"]) + int(sample["a2"])) == 2)

    def test_or(self):
        hybridKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(
            weightedFormulas={},
            facts={"constraint1": ["or", "a1", "a2"]}
        ))
        self.assertEqual(0,
                         hybridKB.ask(["and", ["not", "a1"], ["not", "a2"]]))

        for rep in range(sampleRepetition):
            sample = hybridKB.forward_sample(["a1", "a2"])
            self.assertTrue((int(sample["a1"]) + int(sample["a2"])) >= 1)

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2"])
            self.assertTrue((int(sample["a1"]) + int(sample["a2"])) >= 1)

    def test_xor(self):
        hybridKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(
            weightedFormulas={},
            facts={"constraint1": ["xor", "a1", "a2"]}
        ))
        self.assertEqual(0,
                         hybridKB.ask(["and", "a1", "a2"]))

        for rep in range(sampleRepetition):
            sample = hybridKB.forward_sample(["a1", "a2"])
            self.assertEqual(1 - sample["a1"], sample["a2"])

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2"])
            self.assertEqual(1 - sample["a1"], sample["a2"])

    def test_eq(self):
        hybridKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(
            weightedFormulas={},
            facts={"constraint1": ["eq", "a1", "a2"]}
        ))
        self.assertEqual(0,
                         hybridKB.ask(["and", "a1", ["not", "a2"]]))

        for rep in range(sampleRepetition):
            sample = hybridKB.forward_sample(["a1", "a2"])
            self.assertEqual(sample["a1"], sample["a2"])

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2"])
            self.assertEqual(sample["a1"], sample["a2"])

    def test_imp(self):
        hybridKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(
            weightedFormulas={},
            facts={"constraint1": ["imp", "a1", "a2"]}
        ))
        self.assertEqual(0,
                         hybridKB.ask(["and", "a1", ["not", "a2"]]))

        for rep in range(sampleRepetition):
            sample = hybridKB.forward_sample(["a1", "a2"])
            self.assertEqual(0, int(sample["a1"]) - int(sample["a1"]) * int(sample["a2"]))

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2"])
            self.assertEqual(0, int(sample["a1"]) - int(sample["a1"]) * int(sample["a2"]))

    ##
    def test_unseen_atoms(self):
        hybridKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(
            weightedFormulas={"f1": ["a1", 2]},
            facts={"constraint1": ["imp", "a1", "a2"]}
        ))
        self.assertEqual(3, len(hybridKB.forward_sample(["a3", "a4", "a1"])))
        self.assertEqual(3, len(hybridKB.forward_sample(["fun1", "fun4", "fun5"])))

    def test_evidence_evaluation(self):
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulas={"f1": ["a1", 2]},
            facts={"constraint1": ["imp", "a1", "a2"]}
        )
        entailedDict = knowledge.KnowledgePropagator(hybridKB, evidenceDict={"a1": 0, "a2": 1}).evaluate()
        self.assertTrue(entailedDict["a1"] == 0)
        self.assertTrue(entailedDict["a2"] == 1)
        self.assertTrue(entailedDict["(imp_a1_a2)"] == 1)

    def test_categorical_constraint(self):
        hybridKB = knowledge.InferenceProvider(knowledge.HybridKnowledgeBase(
            weightedFormulas={"f1": ["imp", "a1", "a2", 10]},
            facts={"f2": "a4"},
            categoricalConstraints={"c1": ["a1", "a2", "a3"]}
        ))
        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2", "a3"])
            self.assertTrue(int(sample["a1"]) + int(sample["a2"]) + int(sample["a3"]) == 1)
        for rep in range(sampleRepetition):
            sample = hybridKB.forward_sample(["a1", "a2", "a3"])
            self.assertTrue(int(sample["a1"]) + int(sample["a2"]) + int(sample["a3"]) == 1)


if __name__ == "__main__":
    unittest.main()
