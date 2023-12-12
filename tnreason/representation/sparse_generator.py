import tnreason.representation.ttl_to_csv as ttoc
import tnreason.representation.csv_to_ttl as ctot
import tnreason.representation.sampledf_to_factdf as stof
import tnreason.representation.sampledf_to_pairdf as stop

import tnreason.logic.sparse_cc as scc

import rdflib
import pandas as pd
import numpy as np
import scipy.sparse as scs

class SparseCoreGenerator:
    def __init__(self):
        self.factDf = None
        self.pairDf = None

    def load_ttl(self,ttlPath):
        if self.factDf is None:
            self.factDf = ttoc.generate_csv_list(ttlPath)
        else:
            self.factDf = ttoc.extend_csv_list(self.factDf, ttlPath)

    def load_factDf(self, factDf):
        if self.factDf is None:
            self.factDf = factDf
        else:
            self.factDf = pd.concatenate(self.factDf,factDf)

    def calculate_pairDf(self,extractionQuery):
        ## Need to implement extractionQuery - via ctot and rdflib?
        pass

    def load_sampleDf(self,sampleDf):
        rows = []
        cols = []
        for i, row in sampleDf.iterrows():
            for j, col in enumerate(sampleDf.columns):
                if row[col] == 1:
                    rows.append(i)
                    cols.append(j)
        self.repMatrix = scs.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(sampleDf.shape[0],sampleDf.shape[1]))

        self.dataIndex = sampleDf.index
        self.axiomIndex = list(sampleDf.columns)

    def generate_sparseAxiomCore(self,axiomList=None,axiomColor="var"):
        if axiomList == None:
            return scc.SparseCoordinateCore(core_values=self.repMatrix, core_colors=["j",axiomColor])
        else:
            axiomPositions = [self.axiomIndex.index(axiom) for axiom in axiomList]

            rows = []
            cols = []

            for i,j,v in zip(self.repMatrix.row, self.repMatrix.col, self.repMatrix.data):
                if j in axiomPositions:
                    rows.append(i)
                    cols.append(j)

            return scc.SparseCoordinateCore(core_values=scs.coo_matrix((np.ones(len(rows)),(rows, cols)), shape=self.repMatrix.shape),
                                            core_colors=["j", axiomColor])


if __name__ == "__main__":
    loader = SparseCoreGenerator()
    sampleDf = pd.read_csv("./demonstration/generation/synthetic_test_data/generated_sampleDf.csv")
    loader.load_sampleDf(sampleDf)

    sparsecore = loader.generate_sparseAxiomCore(["Moebel(z)","Bautischlerei(y)"])
    print(sparsecore.values)
