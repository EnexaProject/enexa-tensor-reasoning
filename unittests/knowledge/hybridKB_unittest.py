import unittest

from tnreason import knowledge


class HybridKBTest(unittest.TestCase):
    def test_is_satisfiable(self):
        self.assertTrue(knowledge.HybridKnowledgeBase(
            weightedFormulasDict={"e": ["a1", 2]},
            factsList=["a1", ["a1", "imp", "a2"]]).is_satisfiable())

    def test_satisfiability2(self):
        with self.assertRaises(ValueError, msg="The initialized Knowledge Base is inconsistent!"):
            knowledge.HybridKnowledgeBase(factsList=["a1", ["not", "a1"]])

    def test_ask_constraint_entailed(self):
        self.assertEquals("entailed",
                          knowledge.HybridKnowledgeBase(weightedFormulasDict={"e": ["a1", 2]},
                                                        factsList=["a1"]).ask_constraint("a1")
                          )

    def test_ask_constraint_contradicted(self):
        self.assertEquals("contradicting",
                          knowledge.HybridKnowledgeBase(
                              weightedFormulasDict={"e": [[["a1", "eq", "a2"], "imp", ["a3", "xor", "a1"]], 2]},
        factsList = [["a1", "and", "a2"]]).ask_constraint(
            ["not", "a1"])
        )

    def test_map_query(self):
        self.assertEquals({"a1" : 1, "a2": 0},
                          knowledge.HybridKnowledgeBase(
                              weightedFormulasDict={"e": [[["a1", "eq", "a2"], "imp", ["a3", "xor", "a1"]], 2]},
        factsList = ["a1", ["not","a2"]]).map_query(["a1", "a2"], evidenceDict={"a3":1})
        )


if __name__ == "__main__":
    unittest.main()


