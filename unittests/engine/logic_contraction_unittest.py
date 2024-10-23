import unittest

from tnreason import engine

from tnreason import encoding

methodList = [{"coreType": "NumpyCore", "contractionMethod": "NumpyEinsum"},
              {"coreType": "PolynomialCore", "contractionMethod": "PolynomialContractor"},
              {"coreType": "PandasCore", "contractionMethod": "PolynomialContractor"}
              ]


class TensorLogicTest(unittest.TestCase):
    def test_and(self):

        for method in methodList:
            cores = encoding.create_formulas_cores({"f1": ["and", "a", "b"]}, coreType=method["coreType"])
            contractionResult = engine.contract(coreDict=cores, openColors=["a"], method=method["contractionMethod"])

            self.assertEqual(contractionResult[0], 0)
            self.assertEqual(contractionResult[1], 1)

    def test_and_not(self):

        for method in methodList:
            cores = encoding.create_formulas_cores({"f1": ["and", ["not", "a"], ["or", "b", "c"]]},
                                                   coreType=method["coreType"])
            contractionResult = engine.contract(coreDict=cores, openColors=["a"], method=method["contractionMethod"])

            self.assertEqual(contractionResult[0], 3)
            self.assertEqual(contractionResult[1], 0)

    def test_imp(self):

        for method in methodList:
            cores = encoding.create_formulas_cores({"a": ["imp", "a", "b"]}, coreType=method["coreType"])
            contractionResult = engine.contract(coreDict=cores, openColors=["a", "b"],
                                                method=method["contractionMethod"])
            contractionResult.reorder_colors(["a", "b"])

            self.assertEqual(contractionResult[1, 0], 0)
            self.assertEqual(contractionResult[0, 1], 1)
            self.assertEqual(contractionResult[0, 0], 1)
            self.assertEqual(contractionResult[1, 1], 1)

    def test_eq(self):

        for method in methodList:
            cores = encoding.create_formulas_cores({"a": ["and", ["eq", "a", "b"], ["not", "c"]]},
                                                   coreType=method["coreType"])
            contractionResult = engine.contract(coreDict=cores, openColors=["a", "b"],
                                                method=method["contractionMethod"])
            contractionResult.reorder_colors(["a", "b"])

            self.assertEqual(contractionResult[1, 0], 0)
            self.assertEqual(contractionResult[0, 1], 0)
            self.assertEqual(contractionResult[0, 0], 1)
            self.assertEqual(contractionResult[1, 1], 1)

    def test_xor(self):

        for method in methodList:
            cores = encoding.create_formulas_cores({"xor": ["and", "c1", ["xor", "a", "b"]]},
                                                   coreType=method["coreType"])
            contractionResult = engine.contract(coreDict=cores, openColors=["a", "b"],
                                                method=method["contractionMethod"])
            contractionResult.reorder_colors(["a", "b"])

            self.assertEqual(contractionResult[1, 0], 1)
            self.assertEqual(contractionResult[0, 1], 1)
            self.assertEqual(contractionResult[0, 0], 0)
            self.assertEqual(contractionResult[1, 1], 0)

    def test_disconnected_and(self):

        for method in methodList:
            cores0 = encoding.create_formulas_cores({"xor": ["and", "a", "b"]}, coreType=method["coreType"])
            cores1 = encoding.create_formulas_cores({"f1": ["and", "a", ["not", "c_2"]],
                                                     "f2": "b"}, coreType=method["coreType"])
            result0 = engine.contract(coreDict=cores0, openColors=["a", "b"], method=method["contractionMethod"])
            result0.reorder_colors(["a", "b"])

            result1 = engine.contract(coreDict=cores1, openColors=["a", "b"], method=method["contractionMethod"])
            result1.reorder_colors(["a", "b"])

            for i in range(2):
                for j in range(2):
                    self.assertEqual(result0[i, j], result1[i, j])
