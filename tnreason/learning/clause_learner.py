import tnreason.model.formula_tensors as ft

import tnreason.logic.coordinate_calculus as cc
import numpy as np

import tnreason.contraction.core_contractor as coc
import tnreason.optimization.alternating_mle as amle


class ClauseLearnerMLE(amle.MLEBase):
    ## ! self.superPosedFormulaTensor is substracted!
    def compute_clause_gradient(self, tboVariableKey):
        old_data_gradient = self.contract_data_gradient(tboVariableKey)

        data_correction_contractor = coc.CoreContractor({key: self.superposedFormulaTensor.parameterCoresDict[key] for key in
                                                   self.superposedFormulaTensor.parameterCoresDict if
                                                   key != tboVariableKey})
        ## PROBLEM WHEN VARIABLECORES ARE DISCONNECTED! General bug?
        data_correction_contractor.visualize()
        data_correction_core = data_correction_contractor.contract()

        data_gradient = data_correction_core.sum_with(old_data_gradient.multiply(-1))
        partition_gradient = self.contract_clause_partition_gradient(tboVariableKey)  #
        return partition_gradient.sum_with(data_gradient.multiply(-1))

    def contract_clause_partition_gradient(self, tboVariableKey):
        variablesSum = coc.CoreContractor(self.superposedFormulaTensor.parameterCoresDict,
                                          openColors=[]).contract().values

        variablesExpFactor = coc.CoreContractor(self.superposedFormulaTensor.get_cores(),
                                                openColors=self.candidatesAtomsList).contract(
            optimizationMethod="GreedyHeuristic").multiply(-1).exponentiate().multiply(np.exp(variablesSum))

        contractionDict = {**self.fixedFormulaTensors.all_cores(),
                           "variablesExpFactor": variablesExpFactor,
                           **self.superposedFormulaTensor.get_cores(
                               parameterExceptionKeys=[tboVariableKey])}

        partitionFunction = coc.CoreContractor({**self.fixedFormulaTensors.get_cores(headType="expFactor"),
                                                "variablesExpFactor": variablesExpFactor})

        return coc.CoreContractor(contractionDict, openColors=self.superposedFormulaTensor.parameterCoresDict[
            tboVariableKey].colors).contract(optimizationMethod="GreedyHeuristic").multiply(
            1 / partitionFunction)


def create_skeletonExpression(negLiterals, posLiterals):
    if len(negLiterals) > 0:
        expression = negLiterals[0]
        for negLiteral in negLiterals[1:]:
            expression = [expression, "and", negLiteral]
    else:
        ## if negLiterals empty
        expression = ["not", posLiterals[0]]
        for posLiteral in posLiterals[1:]:
            expression = [expression, "and", ["not", posLiteral]]
        return expression
    for posLiteral in posLiterals:
        expression = [expression, "and", ["not", posLiteral]]
    return expression
