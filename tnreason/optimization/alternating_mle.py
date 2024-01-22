import tnreason.contraction.bc_contraction_generation as bcg
import tnreason.contraction.core_contractor as coc

import tnreason.model.formula_tensors as ft
import tnreason.model.tensor_model as tm
import tnreason.model.entropies as ent

import tnreason.logic.coordinate_calculus as cc
import tnreason.logic.expression_generation as eg

import numpy as np


class MLEBase:
    def __init__(self, skeletonExpression, candidatesDict, variableCoresDict=None, learnedFormulaDict={},
                 sampleDf=None):
        if variableCoresDict is None:
            variableCoresDict = {
                key: cc.CoordinateCore(np.random.random(size=len(candidatesDict[key])), [key], key + "_varCore") for key
                in candidatesDict}
        self.superposedFormulaTensor = ft.SuperposedFormulaTensor(skeletonExpression, candidatesDict,
                                                                  parameterCoresDict=variableCoresDict)

        self.learnedFormulaDict = learnedFormulaDict
        self.fixedFormulaTensors = tm.TensorRepresentation(learnedFormulaDict, headType="weightedTruthEvaluation")
        if sampleDf is not None:
            self.load_sampleDf(sampleDf)

        self.skeletonExpression = skeletonExpression
        self.candidatesDict = candidatesDict
        self.candidatesAtomsList = []
        for candidatesKey in self.candidatesDict:
            for atom in self.candidatesDict[candidatesKey]:
                if atom not in self.candidatesAtomsList:
                    self.candidatesAtomsList.append(atom)

    def load_sampleDf(self, sampleDf):
        self.dataTensor = ft.DataTensor(sampleDf)
        self.compute_fixedCores_likelihoodCorrection()

    def create_exponentiated_variables(self):
        ## Creates exponentiated factor to the variables tbo
        self.variablesExpFactor = coc.CoreContractor(self.superposedFormulaTensor.get_cores(),
                                                     openColors=self.candidatesAtomsList).contract(
            optimizationMethod="GreedyHeuristic").exponentiate()

        ## To do: Think of removing this bottleneck: Constract the current variables to a single core instead of exp TN!

    def contract_partition_gradient(self, tboVariableKey):
        self.create_exponentiated_variables()
        contractionDict = {**self.fixedFormulaTensors.all_cores(),
                           "variablesExpFactor": self.variablesExpFactor,
                           **self.superposedFormulaTensor.get_cores(
                               parameterExceptionKeys=[tboVariableKey])}
        return coc.CoreContractor(contractionDict, openColors=self.superposedFormulaTensor.parameterCoresDict[
            tboVariableKey].colors).contract(optimizationMethod="GreedyHeuristic").multiply(
            1 / self.contract_partition(reDoExpVariables=False))

    def contract_partition(self, visualize=False, reDoExpVariables=True):
        if reDoExpVariables:
            self.create_exponentiated_variables()
        contractor = coc.CoreContractor({  # **self.fixedFormulaTensors.all_cores(),
            **self.fixedFormulaTensors.get_cores(headType="expFactor"),
            "variablesExpFactor": self.variablesExpFactor})
        if visualize:
            contractor.visualize(title="Partition Function")
        return contractor.contract().values

    def contract_data_gradient(self, tboVariableKey):
        contractor = coc.CoreContractor(
            {**self.superposedFormulaTensor.get_cores(parameterExceptionKeys=[tboVariableKey]),
             **self.dataTensor.get_cores()
             },
            openColors=self.superposedFormulaTensor.parameterCoresDict[
                tboVariableKey].colors)
        return contractor.contract().multiply(1 / self.dataTensor.dataNum)

    ## Compute Estimation Metric: Log likelihood (= negative empirical cross entropy)
    def compute_likelihood(self, visualize=False):
        contractor = coc.CoreContractor({**self.superposedFormulaTensor.get_cores(),
                                         **self.dataTensor.get_cores()})
        if visualize:
            contractor.visualize(title="Likelihood VariableCores")

        return contractor.contract().values / (self.dataTensor.dataNum) - np.log(
            self.contract_partition()) + self.formulaCorrectionTerm

    def compute_fixedCores_likelihoodCorrection(self):
        self.formulaCorrectionTerm = np.sum(
            [coc.CoreContractor({**self.fixedFormulaTensors.get_cores([formulaKey], headType="weightedTruthEvaluation"),
                                 **self.dataTensor.get_cores()}).contract().values
             for formulaKey in self.learnedFormulaDict]) / self.dataTensor.dataNum

    def compute_empirical_KL_divergence(self, sampleDf):
        return -self.compute_likelihood() - ent.empirical_shannon_entropy(sampleDf, self.candidatesAtomsList)

    ## Optimization preparation
    def random_initialize_variableCoresDict(self):
        self.superposedFormulaTensor.random_initialize_parameterCoresDict()

    ## Output of the solution
    def get_solution_expression(self):
        return eg.replace_atoms(self.skeletonExpression,
                                self.superposedFormulaTensor.get_largest_weight_as_solutionMap())


class GradientDescentMLE(MLEBase):

    def compute_gradient(self, tboVariableKey):
        partition_gradient = self.contract_partition_gradient(tboVariableKey)
        data_gradient = self.contract_data_gradient(tboVariableKey)
        return partition_gradient.sum_with(data_gradient.multiply(-1))

    def descent_step(self, tboVariableKey, stepWidth=1):
        self.superposedFormulaTensor.parameterCoresDict[tboVariableKey] = \
            self.superposedFormulaTensor.parameterCoresDict[tboVariableKey].sum_with(
                self.compute_gradient(tboVariableKey).multiply(-1 * stepWidth)
            )

    def alternating_gradient_descent(self, sweepNum, stepWidth, computeLikelihood=True, verbose=True):
        safeLikeLihood = -2.2250738585072014e308  # The smallest float
        likelihoods = np.empty(shape=(sweepNum, len(self.superposedFormulaTensor.parameterCoresDict)))
        for sweep in range(sweepNum):
            for i, variableKey in enumerate(self.superposedFormulaTensor.parameterCoresDict):
                self.descent_step(variableKey, stepWidth)
                if computeLikelihood:
                    likelihoods[sweep, i] = self.compute_likelihood()
                    if likelihoods[sweep, i] < safeLikeLihood and verbose:
                        print(
                            "Warning: Likelihood increased on variableCore {} in sweep {}.".format(variableKey, sweep))
                    if likelihoods[sweep, i] > 0:
                        print("Warning: Positive Likelihood on variableCore {} in sweep {}.".format(variableKey, sweep))
                    safeLikeLihood = likelihoods[sweep, i]
        return likelihoods


## Does not yet make use of the superposedFormulaTensor -> Still on pure conjunctions
class AlternatingNewtonMLE(MLEBase):
    def create_atom_selectors(self):
        ## Creates atom selector cores selecting atom activation
        self.atomSelectorDict = {}
        for placeHolderKey in self.candidatesDict:
            for i, atomKey in enumerate(self.candidatesDict[placeHolderKey]):
                coreValues = np.ones(shape=(len(self.candidatesDict[placeHolderKey]), 2))
                coreValues[i, 0] = 0
                self.atomSelectorDict[placeHolderKey + "_enumeratedWorlds_" + atomKey] = cc.CoordinateCore(
                    coreValues, [placeHolderKey, atomKey], placeHolderKey + "_enumeratedWorlds_" + atomKey)

    def create_variable_gradient(self, tboVariableKey):
        return {key: self.variableCoresDict[key] for key in self.variableCoresDict if key != tboVariableKey}

    def contract_double_gradient_exponential(self, tboVariableKey):

        self.formulaCoreDict = bcg.generate_formulaCoreDict(self.learnedFormulaDict)

        contractionDict = {**self.formulaCoreDict, "variablesExpFactor": self.variablesExpFactor,
                           **self.create_variable_gradient(tboVariableKey),
                           **self.atomSelectorDict,
                           **copy_CoreDict(self.create_variable_gradient(tboVariableKey), suffix="out"),
                           **copy_CoreDict(self.atomSelectorDict, suffix="out",
                                           exceptionColors=self.candidatesAtomsList),
                           }
        openColors = self.variableCoresDict[tboVariableKey].colors
        openColors = openColors + [color + "_out" for color in openColors]
        contractor = coc.CoreContractor(contractionDict, openColors=openColors)
        return contractor.contract(optimizationMethod="GreedyHeuristic")

    ## Algorithm
    def alternating_newton(self, sweepNum=10, dampFactor=0.001, monotoneousCondition=True, regParameter=0,
                           verbose=True):
        self.create_atom_selectors()
        self.variableCoresDict = self.superposedFormulaTensor.parameterCoresDict

        self.likelihood = self.compute_likelihood()
        likelihoods = np.empty(shape=(sweepNum, len(self.variableCoresDict)))
        for sweep in range(sweepNum):
            print("### SWEEP {} ###".format(sweep))
            for i, variableKey in enumerate(self.variableCoresDict):
                self.newton_step(variableKey, dampFactor=dampFactor, monotoneusCondition=monotoneousCondition,
                                 regParameter=regParameter, verbose=verbose)
                likelihoods[sweep, i] = self.likelihood
        return likelihoods

    def newton_step(self, tboVariableKey, dampFactor, monotoneusCondition, regParameter, verbose=True):
        expGradient = self.contract_partition_gradient(tboVariableKey)

        vector = expGradient.sum_with(
            self.contract_data_gradient(tboVariableKey).multiply(-1 / self.contract_partition()))

        ## Operator
        doubleExpGradient = self.contract_double_gradient_exponential(tboVariableKey).multiply(-1)

        empOperator = expGradient.clone()
        empOperator.colors = [color + "_out" for color in empOperator.colors]
        empOperator = empOperator.compute_and(self.contract_data_gradient(tboVariableKey))

        operator = empOperator.sum_with(doubleExpGradient)

        ## Flatten
        outColors = [color for color in operator.colors if color.endswith("_out")]
        inColors = [color for color in operator.colors if color not in outColors]

        outshape = [operator.values.shape[i] for i, color in enumerate(operator.colors) if color in outColors]
        inshape = [operator.values.shape[i] for i, color in enumerate(operator.colors) if color in inColors]

        outDim = np.prod(outshape)
        inDim = np.prod(inshape)

        operator.reorder_colors(inColors + outColors)
        vector.reorder_colors(inColors)

        flattenedOperatorValues = operator.values.reshape(inDim, outDim) + regParameter * np.eye(inDim)
        flattenedVectorValues = vector.values.reshape(inDim)

        ## Solve Newton
        update, residuals, rank, singular_values = np.linalg.lstsq(flattenedOperatorValues, flattenedVectorValues)

        update.reshape(inshape)
        updateCore = cc.CoordinateCore(update.reshape(inshape), inColors)

        oldCore = self.variableCoresDict[tboVariableKey].clone()

        self.variableCoresDict[tboVariableKey] = self.variableCoresDict[tboVariableKey].sum_with(
            updateCore.multiply(dampFactor))

        if monotoneusCondition:
            likelihood = self.compute_likelihood()
            if likelihood < self.likelihood:
                print("Update of {} is not accepted.".format(tboVariableKey))
                self.variableCoresDict[tboVariableKey] = oldCore
            else:
                self.likelihood = likelihood

        if verbose:
            print("Update Norm", np.linalg.norm(update), "Operator Condition", np.linalg.cond(flattenedOperatorValues))
            print(self.compute_likelihood())


def copy_CoreDict(tbdDict, suffix="", exceptionColors=[]):
    doubleDict = {}
    for key in tbdDict:
        oldCore = tbdDict[key]
        newColors = []
        for color in oldCore.colors:
            if color in exceptionColors:
                newColors.append(color)
            else:
                newColors.append(color + "_" + suffix)
        doubleDict[key + "_" + suffix] = cc.CoordinateCore(oldCore.values, newColors)
    return doubleDict


if __name__ == "__main__":
    from tnreason.logic import coordinate_calculus as cc

    atomDict = {
        "a": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l1", "y", "z"], name="a"),
        "b": cc.CoordinateCore(np.random.binomial(n=1, p=0.8, size=(10, 7, 5)), ["l2", "q", "z"], name="b"),
        "c": cc.CoordinateCore(np.random.binomial(n=1, p=0.4, size=(10, 7, 5)), ["l3", "q", "z"], name="c"),
    }

    skeletonExpression = ["P1", "and", "P2"]
    candidatesDict = {"P1": list(atomDict.keys()),
                      "P2": list(atomDict.keys()),
                      }

    variableCoresDict = {
        "v1": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P1", "H1"]),
        "v2": cc.CoordinateCore(np.zeros(shape=(3, 2)), ["P2", "H1"]),
    }

    learnedFormulaDict = {
        "f0": ["b", 10],
        "f1": [["not", ["a", "and", "b"]], 5],
        "f2": ["c", 2]
    }

    import tnreason.model.generate_test_data as gtd

    sampleDf = gtd.generate_sampleDf(learnedFormulaDict, 100)

    optimizer = AlternatingNewtonMLE(skeletonExpression, candidatesDict, variableCoresDict, {}, sampleDf=sampleDf)

    optimizer.random_initialize_variableCoresDict()

    optimizer.create_exponentiated_variables()

    optimizer.alternating_newton(10)
