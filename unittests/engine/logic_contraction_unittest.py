import unittest

from tnreason import engine

from tnreason import encoding

# testMethods = ["NumpyEinsum", "NumpyEinsum"]
testMethods = ["PgmpyVariableEliminator", "NumpyEinsum", "TensorFlowEinsum", "TorchEinsum"]


class TensorLogicTest(unittest.TestCase):
    def test_and(self):
        cores = encoding.create_formulas_cores({"f1": ["a", "and", "b"]})
        for method in testMethods:
            contractionResult = engine.contract(coreDict=cores, openColors=["a"], method=method)

            self.assertEquals(contractionResult.values[0], 0)
            self.assertEquals(contractionResult.values[1], 1)

    def test_and_not(self):
        cores = encoding.create_formulas_cores({"f1": [["not", "a"], "and", ["b", "or", "c"]]})
        for method in testMethods:
            contractionResult = engine.contract(coreDict=cores, openColors=["a"], method=method)

            self.assertEquals(contractionResult.values[0], 3)
            self.assertEquals(contractionResult.values[1], 0)

    def test_imp(self):
        cores = encoding.create_formulas_cores({"a":["a", "imp", "b"]})
        for method in testMethods:
            contractionResult = engine.contract(coreDict=cores, openColors=["a", "b"], method=method)
            contractionResult.reorder_colors(["a", "b"])

            self.assertEquals(contractionResult.values[1, 0], 0)
            self.assertEquals(contractionResult.values[0, 1], 1)
            self.assertEquals(contractionResult.values[0, 0], 1)
            self.assertEquals(contractionResult.values[1, 1], 1)

    def test_eq(self):
        cores = encoding.create_formulas_cores({"a":[["a", "eq", "b"], "and", ["not", "c"]]})
        for method in testMethods:
            contractionResult = engine.contract(coreDict=cores, openColors=["a", "b"], method=method)
            contractionResult.reorder_colors(["a", "b"])

            self.assertEquals(contractionResult.values[1, 0], 0)
            self.assertEquals(contractionResult.values[0, 1], 0)
            self.assertEquals(contractionResult.values[0, 0], 1)
            self.assertEquals(contractionResult.values[1, 1], 1)

    def test_xor(self):
        cores = encoding.create_formulas_cores({"xor":["c1", "and", ["a", "xor", "b"]]})
        for method in testMethods:
            contractionResult = engine.contract(coreDict=cores, openColors=["a", "b"], method=method)
            contractionResult.reorder_colors(["a", "b"])

            self.assertEquals(contractionResult.values[1, 0], 1)
            self.assertEquals(contractionResult.values[0, 1], 1)
            self.assertEquals(contractionResult.values[0, 0], 0)
            self.assertEquals(contractionResult.values[1, 1], 0)

    def test_disconnected_and(self):
        cores0 = encoding.create_formulas_cores({"xor": ["a", "and", "b"]})
        cores1 = encoding.create_formulas_cores({"f1":["a", "and", ["not", "c_2"]],
                                                 "f2":"b"})

        for method in testMethods:
            result0 = engine.contract(coreDict=cores0, openColors=["a", "b"], method=method)
            result0.reorder_colors(["a", "b"])

            result1 = engine.contract(coreDict=cores1, openColors=["a", "b"], method=method)
            result1.reorder_colors(["a", "b"])

            for i in range(2):
                for j in range(2):
                    self.assertEquals(result0.values[i, j], result1.values[i, j])
