import unittest

from tnreason import algorithms
from tnreason import encoding


rules = {
    "f1" : ["imp","a1","a2"],
}

class FCTest(unittest.TestCase):
    def test_modus_ponens(self):
        preEvidence = {
            "a1" : 1
        }

        propagator = algorithms.ConstraintPropagator(
            {**encoding.create_formulas_cores(rules),
             **encoding.create_evidence_cores(preEvidence)},
            verbose=False
        )
        propagator.propagate_cores()
        evidenceDict, multipleAssignmentColors, redundantCores, remainingCores = propagator.find_evidence_and_redundant_cores()

        self.assertTrue(evidenceDict["a2"] == 1)

    def test_refutation(self):
        preEvidence = {
            "a2" : 0
        }

        propagator = algorithms.ConstraintPropagator(
            {**encoding.create_formulas_cores(rules),
             **encoding.create_evidence_cores(preEvidence)},
            verbose=False
        )
        propagator.propagate_cores()
        evidenceDict, multipleAssignmentColors, redundantCores, remainingCores = propagator.find_evidence_and_redundant_cores()

        self.assertTrue(evidenceDict["a1"] == 0)

if __name__ == "__main__":
    unittest.main()
