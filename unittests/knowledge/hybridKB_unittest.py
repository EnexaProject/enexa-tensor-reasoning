import unittest

from tnreason import knowledge

sampleRepetition = 50


class HybridKBTest(unittest.TestCase):
    def test_is_satisfiable(self):
        self.assertTrue(knowledge.HybridKnowledgeBase(
            weightedFormulasDict={"e": ["a1", 2]},
            factsDict={"c1": "a1",
                       "c2": ["a1", "imp", "a2"]}).is_satisfiable())

    def test_satisfiability2(self):
        with self.assertRaises(ValueError, msg="The initialized Knowledge Base is inconsistent!"):
            knowledge.HybridKnowledgeBase(factsDict={"c1": "a1", "c2": ["not", "a1"]})

    def test_ask_constraint_entailed(self):
        self.assertEquals("entailed",
                          knowledge.HybridKnowledgeBase(weightedFormulasDict={"e": ["a1", 2]},
                                                        factsDict={"c1": "a1"}).ask_constraint("a1")
                          )

    def test_ask_constraint_contradicted(self):
        self.assertEquals("contradicting",
                          knowledge.HybridKnowledgeBase(
                              weightedFormulasDict={"e": [[["a1", "eq", "a2"], "imp", ["a3", "xor", "a1"]], 2]},
                              factsDict={"c1": ["a1", "and", "a2"]}).ask_constraint(
                              ["not", "a1"])
                          )

    def test_map_query(self):
        self.assertEquals({"a1": 1, "a2": 0},
                          knowledge.HybridKnowledgeBase(
                              weightedFormulasDict={"e": [[["a1", "eq", "a2"], "imp", ["a3", "xor", "a1"]], 2]},
                              factsDict={"c1": "a1",
                                         "c2": ["not", "a2"]}).exact_map_query(["a1", "a2"], evidenceDict={"a3": 1})
                          )

    def test_empty_dicts(self):
        self.assertEquals(1,
                          knowledge.HybridKnowledgeBase(
                              weightedFormulasDict={}, factsDict={}).query(["a1"], evidenceDict={"a1": 1}).values[1])
        self.assertEquals(0.5,
                          knowledge.HybridKnowledgeBase(
                              weightedFormulasDict={}, factsDict={}).query(["a1"], evidenceDict={}).values[1])
        self.assertEquals(0.125,
                          knowledge.HybridKnowledgeBase(
                              weightedFormulasDict={}, factsDict={}).query(["a1", "a3", "a2"], evidenceDict={}).values[
                              1, 0, 1])

    ## Sampling on facts tests
    def test_not(self):
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={},
            factsDict={"constraint1": ["not", "a1"]}
        )
        self.assertEquals(0,
                          hybridKB.ask("a1"))

        for rep in range(sampleRepetition):
            sample = hybridKB.annealed_map_query(["a1"])
            self.assertEquals(0, sample["a1"])

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1"])
            self.assertEquals(0, sample["a1"])

    def test_and(self):
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={"f1": ["a1", 1]},
            factsDict={"constraint1": ["a1", "and", "a2"]}
        )
        self.assertEquals(0,
                          hybridKB.ask(["not", "a1"]))

        for rep in range(sampleRepetition):
            sample = hybridKB.annealed_map_query(["a1", "a2"])
            self.assertTrue((int(sample["a1"]) + int(sample["a2"])) == 2)

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2"])
            self.assertTrue((int(sample["a1"]) + int(sample["a2"])) == 2)

    def test_or(self):
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={},
            factsDict={"constraint1": ["a1", "or", "a2"]}
        )
        self.assertEquals(0,
                          hybridKB.ask([["not", "a1"], "and", ["not", "a2"]]))

        for rep in range(sampleRepetition):
            sample = hybridKB.annealed_map_query(["a1", "a2"])
            self.assertTrue((int(sample["a1"]) + int(sample["a2"])) >= 1)

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2"])
            self.assertTrue((int(sample["a1"]) + int(sample["a2"])) >= 1)

    def test_xor(self):
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={},
            factsDict={"constraint1": ["a1", "xor", "a2"]}
        )
        self.assertEquals(0,
                          hybridKB.ask(["a1", "and", "a2"]))

        for rep in range(sampleRepetition):
            sample = hybridKB.annealed_map_query(["a1", "a2"])
            self.assertEquals(1 - sample["a1"], sample["a2"])

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2"])
            self.assertEquals(1 - sample["a1"], sample["a2"])

    def test_eq(self):
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={},
            factsDict={"constraint1": ["a1", "eq", "a2"]}
        )
        self.assertEquals(0,
                          hybridKB.ask(["a1", "and", ["not", "a2"]]))

        for rep in range(sampleRepetition):
            sample = hybridKB.annealed_map_query(["a1", "a2"])
            self.assertEquals(sample["a1"], sample["a2"])

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2"])
            self.assertEquals(sample["a1"], sample["a2"])

    def test_imp(self):
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={},
            factsDict={"constraint1": ["a1", "imp", "a2"]}
        )
        self.assertEquals(0,
                          hybridKB.ask(["a1", "and", ["not", "a2"]]))

        for rep in range(sampleRepetition):
            sample = hybridKB.annealed_map_query(["a1", "a2"])
            self.assertEquals(0, int(sample["a1"]) - int(sample["a1"]) * int(sample["a2"]))

        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2"])
            self.assertEquals(0, int(sample["a1"]) - int(sample["a1"]) * int(sample["a2"]))

    ##
    def test_unseen_atoms(self):
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={"f1": ["a1", 2]},
            factsDict={"constraint1": ["a1", "imp", "a2"]}
        )
        self.assertEquals(3, len(hybridKB.annealed_map_query(["a3", "a4", "a1"])))
        self.assertEquals(3, len(hybridKB.annealed_map_query(["fun1", "fun4", "fun5"])))

    def test_evidence_evaluation(self):
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={"f1": ["a1", 2]},
            factsDict={"constraint1": ["a1", "imp", "a2"]}
        )
        entailed, contradicted, contingent = hybridKB.evaluate_evidence({"a1": 1, "a2": 0})
        self.assertEquals(entailed, ["f1"])
        self.assertEquals(contradicted, ["constraint1"])
        self.assertEquals(contingent, [])

    def test_categorical_constraint(self):
        hybridKB = knowledge.HybridKnowledgeBase(
            weightedFormulasDict={"f1": [["a1", "imp", "a2"], 10]},
            factsDict={"f2": "a4"},
            categoricalConstraintsDict={"c1": ["a1", "a2", "a3"]}
        )
        for rep in range(sampleRepetition):
            sample = hybridKB.exact_map_query(["a1", "a2", "a3"])
            self.assertTrue(int(sample["a1"]) + int(sample["a2"]) + int(sample["a3"]) == 1)
        for rep in range(sampleRepetition):
            sample = hybridKB.annealed_map_query(["a1", "a2", "a3"])
            self.assertTrue(int(sample["a1"]) + int(sample["a2"]) + int(sample["a3"]) == 1)


if __name__ == "__main__":
    unittest.main()
