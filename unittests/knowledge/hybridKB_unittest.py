import unittest

from tnreason import knowledge
import numpy as np


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


if __name__ == "__main__":
    unittest.main()
