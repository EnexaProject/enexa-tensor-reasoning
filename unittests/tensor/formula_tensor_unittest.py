import unittest

from tnreason.tensor import formula_tensors as ft

import pandas as pd

class DataTensorTest(unittest.TestCase):
    def test_positive_entropy(self):
        sampleDf = pd.read_csv("../assets/bbb_generated.csv")
        dataTensor = ft.DataTensor(sampleDf)
        self.assertTrue(dataTensor.compute_shannon_entropy()>0)

