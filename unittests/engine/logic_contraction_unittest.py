import unittest

from tnreason.tensor import formula_tensors as ft
from tnreason import engine


testMethods = ["NumpyEinsum", "NumpyEinsum"]
#testMethods = ["PgmpyVariableEliminator", "PgmpyVariableEliminator", "NumpyEinsum"]

class TensorLogicTest(unittest.TestCase):
    def test_and(self):
        cores = ft.FormulaTensor(["a", "and", "b"]).get_cores()

        for method in testMethods:
            print(cores)
            contractionResult = engine.contract(coreDict=cores, openColors=["a"], method=method)

            self.assertEquals(contractionResult.values[0], 0)
            self.assertEquals(contractionResult.values[1], 1)

    def test_and_not(self):
        cores = ft.FormulaTensor([["not", "a"], "and", ["b", "or", "c"]]).get_cores()

        for method in testMethods:
            contractionResult = engine.contract(coreDict=cores, openColors=["a"], method=method)

            self.assertEquals(contractionResult.values[0], 3)
            self.assertEquals(contractionResult.values[1], 0)

    def test_imp(self):
        cores = ft.FormulaTensor(["a", "imp", "b"]).get_cores()

        for method in testMethods:
            contractionResult = engine.contract(coreDict=cores, openColors=["a", "b"], method=method)
            contractionResult.reorder_colors(["a", "b"])

            self.assertEquals(contractionResult.values[1, 0], 0)
            self.assertEquals(contractionResult.values[0, 1], 1)
            self.assertEquals(contractionResult.values[0, 0], 1)
            self.assertEquals(contractionResult.values[1, 1], 1)

    def test_eq(self):
        cores = ft.FormulaTensor([["a", "eq", "b"], "and", ["not", "c"]]).get_cores()


        for method in testMethods:
            contractionResult = engine.contract(coreDict=cores, openColors=["a", "b"], method=method)
            contractionResult.reorder_colors(["a", "b"])

            self.assertEquals(contractionResult.values[1, 0], 0)
            self.assertEquals(contractionResult.values[0, 1], 0)
            self.assertEquals(contractionResult.values[0, 0], 1)
            self.assertEquals(contractionResult.values[1, 1], 1)

    def test_xor(self):
        cores = ft.FormulaTensor(["c1", "and", ["a", "xor", "b"]]).get_cores()

        for method in testMethods:
            contractionResult = engine.contract(coreDict=cores, openColors=["a", "b"], method=method)
            contractionResult.reorder_colors(["a", "b"])

            self.assertEquals(contractionResult.values[1, 0], 1)
            self.assertEquals(contractionResult.values[0, 1], 1)
            self.assertEquals(contractionResult.values[0, 0], 0)
            self.assertEquals(contractionResult.values[1, 1], 0)

    def test_disconnected_and(self):
        cores0 = ft.FormulaTensor(["a", "and", "b"]).get_cores()
        cores1 = {**ft.FormulaTensor(["a", "and", ["not", "c_2"]]).get_cores(),
                  **ft.FormulaTensor("b").get_cores()}

        for method in testMethods:
            result0 = engine.contract(coreDict=cores0, openColors=["a", "b"], method=method)
            result0.reorder_colors(["a", "b"])

            result1 = engine.contract(coreDict=cores1, openColors=["a", "b"], method=method)
            result1.reorder_colors(["a", "b"])

            for i in range(2):
                for j in range(2):
                    self.assertEquals(result0.values[i, j], result1.values[i, j])
