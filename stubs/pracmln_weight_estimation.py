from tnreason.logic import expression_generation as eg
# from tnreason.logic import expression_calculus as ec
from tnreason.logic import expression_utils as eu

from pracmln import MLN, Database
from pracmln.mlnlearn import MLNLearn
from pracmln.utils import config, locs
from pracmln.utils.project import PRACMLNConfig
from pracmln.utils.config import global_config_filename
import os


class WeightEstimator:
    def __init__(self, solutionList=None):
        self.create_mln(solutionList)
        self.db = Database(self.mln)

    def create_mln(self, solutionList):
        self.mln = MLN(grammar="StandardGrammar",
                       logic="FirstOrderLogic")
        variablesList = eu.get_all_variables(solutionList)
        for variable in variablesList:
            self.mln << variable
        for expression in solutionList:
            self.mln << "0 " + eg.generate_pracmln_formulastring(expression)

    def extend_db(self, factDf):

        for i, row in factDf.iterrows():
            if row["predicate"] == "rdf:type":
                cla = row["object"]
                ind = row["subject"]
                self.db << cla + "(" + ind + ")"
            else:
                ind1 = row["subject"]
                ind2 = row["object"]
                rel = row["predicate"]
                self.db << rel + "(" + ind1 + "," + ind2 + ")"

    def learn_weights(self):
        DEFAULT_CONFIG = os.path.join(locs.user_data, global_config_filename)
        conf = PRACMLNConfig(DEFAULT_CONFIG)

        learner = MLNLearn(conf, mln=self.mln, db=self.db)
        result = learn.run()


if __name__ == "__main__":
    import pandas as pd

    factDf = pd.read_csv("./demonstration/generation/synthetic_test_data/generated_factDf.csv")

    # solutionList = ["bearbeitet(y,x)", "and", "enthaelt(x,z)"]
    # westimator = WeightEstimator(solutionList)
