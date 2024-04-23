import unittest

from tnreason import algorithms
from tnreason import encoding

rules = {
    "f1": ["imp", "a1", "a2"],
}


class FCTest(unittest.TestCase):
    def test_modus_ponens(self):
        preEvidence = {
            "a1": 1
        }

        propagator = algorithms.ConstraintPropagator(
            {**encoding.create_formulas_cores(rules),
             **encoding.create_evidence_cores(preEvidence)},
            verbose=False
        )
        propagator.propagate_cores()
        assignmentDict = propagator.find_assignments()

        self.assertTrue(assignmentDict["a2"] == 1)

    def test_refutation(self):
        preEvidence = {
            "a2": 0
        }

        propagator = algorithms.ConstraintPropagator(
            {**encoding.create_formulas_cores(rules),
             **encoding.create_evidence_cores(preEvidence)},
            verbose=False
        )
        propagator.propagate_cores()
        assignmentDict = propagator.find_assignments()

        self.assertTrue(assignmentDict["a1"] == 0)

        activationCone = propagator.find_variable_cone(["a1"])
        self.assertTrue("a1_domainCore" in activationCone)
        self.assertTrue(len(activationCone) == 1)

    def test_activationCone_pureImp(self):
        propagator = algorithms.ConstraintPropagator(encoding.create_formulas_cores(rules), verbose=False)
        propagator.propagate_cores()
        activationCone = propagator.find_variable_cone(["a1", "a2"])

        self.assertTrue(len(activationCone) == 4)
        self.assertTrue("(imp_a1_a2)_conCore" in activationCone)
        self.assertTrue("a1_domainCore" in activationCone)
        self.assertTrue("a2_domainCore" in activationCone)

    def test_activationCone_andFact(self):
        propagator = algorithms.ConstraintPropagator(encoding.create_formulas_cores({"r1": ["and", "a1", "a2"]}),
                                                     verbose=False)
        propagator.propagate_cores()
        activationCone = propagator.find_variable_cone(["a1", "a2"])

        self.assertTrue(len(activationCone) == 2)
        self.assertTrue("a1_domainCore" in activationCone)
        self.assertTrue("a2_domainCore" in activationCone)


if __name__ == "__main__":
    unittest.main()
