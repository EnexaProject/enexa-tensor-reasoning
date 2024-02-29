import unittest

from tnreason.tensor import formula_tensors as ft
from tnreason import contraction

defaultMethod = "NumpyEinsum"


# defaultMethod = "PgmpyVariableEliminator"

class TensorLogicTest(unittest.TestCase):
    def test_and(self, method=defaultMethod):
        cores = ft.FormulaTensor(["a", "and", "b"]).get_cores()

        contractor = contraction.get_contractor(method)(cores, openColors=["a"])
        contractionResult = contractor.contract()

        self.assertEquals(contractionResult.values[0], 0)
        self.assertEquals(contractionResult.values[1], 1)

    def test_and_not(self, method=defaultMethod):
        cores = ft.FormulaTensor([["not", "a"], "and", ["b", "or", "c"]]).get_cores()

        contractor = contraction.get_contractor(method)(cores, openColors=["a"])
        contractionResult = contractor.contract()

        self.assertEquals(contractionResult.values[0], 3)
        self.assertEquals(contractionResult.values[1], 0)

    def test_imp(self, method=defaultMethod):
        cores = ft.FormulaTensor(["a", "imp", "b"]).get_cores()

        contractor = contraction.get_contractor(method)(cores, openColors=["a", "b"])
        contractionResult = contractor.contract()
        contractionResult.reorder_colors(["a", "b"])

        self.assertEquals(contractionResult.values[1, 0], 0)
        self.assertEquals(contractionResult.values[0, 1], 1)
        self.assertEquals(contractionResult.values[0, 0], 1)
        self.assertEquals(contractionResult.values[1, 1], 1)

    def test_eq(self, method=defaultMethod):
        cores = ft.FormulaTensor([["a", "eq", "b"], "and", ["not", "c"]]).get_cores()

        contractor = contraction.get_contractor(method)(cores, openColors=["a", "b"])
        contractionResult = contractor.contract()
        contractionResult.reorder_colors(["a", "b"])

        self.assertEquals(contractionResult.values[1, 0], 0)
        self.assertEquals(contractionResult.values[0, 1], 0)
        self.assertEquals(contractionResult.values[0, 0], 1)
        self.assertEquals(contractionResult.values[1, 1], 1)

    def test_xor(self, method=defaultMethod):
        cores = ft.FormulaTensor(["c1", "and", ["a", "xor", "b"]]).get_cores()

        contractor = contraction.get_contractor(method)(cores, openColors=["a", "b"])
        contractionResult = contractor.contract()
        contractionResult.reorder_colors(["a", "b"])

        self.assertEquals(contractionResult.values[1, 0], 1)
        self.assertEquals(contractionResult.values[0, 1], 1)
        self.assertEquals(contractionResult.values[0, 0], 0)
        self.assertEquals(contractionResult.values[1, 1], 0)

    def test_disconnected_and(self, method=defaultMethod):
        cores0 = ft.FormulaTensor(["a", "and", "b"]).get_cores()
        cores1 = {**ft.FormulaTensor(["a", "and", ["not", "c_2"]]).get_cores(),
                  **ft.FormulaTensor("b").get_cores()}
        contractor0 = contraction.get_contractor(method)(cores0, openColors=["a", "b"])
        result0 = contractor0.contract()
        result0.reorder_colors(["a", "b"])

        contractor1 = contraction.get_contractor(method)(cores1, openColors=["a", "b"])
        result1 = contractor1.contract()
        result1.reorder_colors(["a", "b"])

        for i in range(2):
            for j in range(2):
                self.assertEquals(result0.values[i, j], result1.values[i, j])
