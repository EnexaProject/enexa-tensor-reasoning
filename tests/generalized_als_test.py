from tnreason.optimization import generalized_als

from tnreason.logic import expression_generation as eg, coordinate_calculus as cc

import numpy as np

if __name__ == "__main__":
    datanum = 10
    legdim = 100

    datacore_1 = np.random.normal(size=(2 ,legdim, datanum))
    datacore_2 = np.random.normal(size=(legdim + 1, datanum, datanum + 1))
    datacore_3 = np.random.normal(size=(legdim + 1, datanum + 1))

    datacores_dict = {
        "1": cc.CoordinateCore(datacore_1, ["l01","l1", "a"]),
        "2": cc.CoordinateCore(datacore_2, ["l2", "a", "b"]),
        "3": cc.CoordinateCore(datacore_3, ["l3", "b"]),
    }

    legvector_1 = np.random.normal(size=(2, legdim))
    legvector_2 = np.random.normal(size=(legdim + 1))
    legvector_3 = np.random.normal(size=(legdim + 1))

    legvectors_dict = {
        "1": cc.CoordinateCore(legvector_1, ["l01","l1"]),
        "2": cc.CoordinateCore(legvector_2, ["l2"]),
        "3": cc.CoordinateCore(legvector_3, ["l3"]),
    }

    y_values = np.random.binomial(n=1,p=0.4,size=(datanum + 1, datanum))
    y_colors = ["b", "a"]
    y = cc.CoordinateCore(y_values, y_colors)

    con_scheme = eg.generate_negated_conjunctions(["1", "3"],["2"])
    print(con_scheme)

    example_als = generalized_als.GeneralizedALS(legvectors_dict, datacores_dict)
    example_als.set_targetCore(y)

    example_als.set_filterCore(y)

    example_als.sweep(10, con_scheme)
    example_als.visualize_residua()
