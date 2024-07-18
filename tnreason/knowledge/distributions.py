from tnreason import encoding
from tnreason import engine

probFormulasKey = "weightedFormulas"
logFormulasKey = "facts"
categoricalsKey = "categoricalConstraints"
evidenceKey = "evidence"

"""
Distributions are Markov Networks with two methods:
    * create_cores(): returning the factor cores
    * get_partition_function(allAtoms): returning the partition function given the atomic variables of interest
"""


class EmpiricalDistribution:
    """
    Inferable (by InferenceProvider) empirical distributions
    """

    def __init__(self, sampleDf, atomKeys=None, interpretation="atomic", dimensionsDict={}):
        """
        * sampleDf: pd.DataFrame containing the samples defining the empirical distributions
        * atomKeys: List of columns of sampleDf to be recognized as atoms
        * interpretation: Specifies the interpretation of the entries of sampleDf
            - "atomic": Variables have dimension 2 and entries in [0,1] are the probability of the atom holding.
            - "categorical": Variables have dimension m specified in dimensionsDict and entries are the certain value of the variable in [m]
        """
        if atomKeys is None:
            atomKeys = list(sampleDf.columns)
        self.atoms = atomKeys
        self.sampleDf = sampleDf
        self.dataNum = sampleDf.values.shape[0]

        self.interpretation = interpretation
        self.dimensionsDict = dimensionsDict

    def __str__(self):
        return "Empirical Distribution with {} samples of atoms {}.".format(self.dataNum, self.atoms)

    def create_cores(self):
        return encoding.create_data_cores(self.sampleDf, atomKeys=self.atoms, interpretation=self.interpretation,
                                          dimensionsDict=self.dimensionsDict)

    def get_partition_function(self, allAtoms=[]):
        unseenAtomNum = len([atom for atom in allAtoms if atom not in self.atoms])
        return (self.dataNum * (2 ** unseenAtomNum))


class HybridKnowledgeBase:
    """
    Inferable (by HybridInferer) Knowledge Base. Generalizes Markov Logic Network by further dedicated cores
    * dimensionDict: Dictionary of dimensions for in the formulas appearing categorical variables
    """

    def __init__(self, weightedFormulas={}, facts={}, categoricalConstraints={}, evidence={}, backCores={},
                 dimensionDict={}):
        self.weightedFormulas = {key: weightedFormulas[key][:-1] + [float(weightedFormulas[key][-1])] for key in
                                 weightedFormulas}
        self.facts = facts
        self.categoricalConstraints = categoricalConstraints
        self.evidence = evidence

        ## Option to add arbitrary factor cores -> Not supported in yaml save/load and atom search, only influenceing create_cores!
        self.backCores = backCores

        self.find_atoms()
        self.dimensionDict=dimensionDict

    def __str__(self):
        outString = "Hybrid Knowledge Base consistent of"
        if self.weightedFormulas:
            outString = outString + "\n######## probabilistic formulas:\n" + "\n".join(
                [encoding.get_formula_color(expression[:-1]) + " with weight " + str(expression[-1]) for expression in
                 self.weightedFormulas.values()])
        if self.facts:
            outString = outString + "\n######## logical formulas:\n" + "\n".join(
                [encoding.get_formula_color(expression) for expression in self.facts.values()])
        if self.categoricalConstraints:
            outString = outString + "\n######## categorical variables:\n" + "\n".join(
                [key + " selecting one of " + " ".join(self.categoricalConstraints[key]) for key in
                 self.categoricalConstraints]
            )
        if self.backCores:
            outString = outString + "\n######## further cores:\n" + "\n".join(list(self.backCores.keys()))
        return outString

    def find_atoms(self):
        """
        Identifies the atoms of the Knowledge Base
        """
        self.atoms = encoding.get_all_atoms(
            {**{key: self.weightedFormulas[key][:-1] for key in self.weightedFormulas},
             **self.facts})
        for constraintKey in self.categoricalConstraints:
            for atom in self.categoricalConstraints[constraintKey]:
                if atom not in self.atoms:
                    self.atoms.append(atom)
        for eKey in self.evidence:
            if eKey not in self.atoms:
                self.atoms.append(eKey)
        self.atoms = list(self.atoms)

    def from_yaml(self, loadPath):
        modelSpec = encoding.load_from_yaml(loadPath)
        if probFormulasKey in modelSpec:
            self.weightedFormulas = modelSpec[probFormulasKey]
        if logFormulasKey in modelSpec:
            self.facts = modelSpec[logFormulasKey]
        if categoricalsKey in modelSpec:
            self.categoricalConstraints = modelSpec[categoricalsKey]
        if evidenceKey in modelSpec:
            self.evidence = modelSpec[evidenceKey]
        self.find_atoms()

    def to_yaml(self, savePath):
        encoding.storage.save_as_yaml({
            probFormulasKey: self.weightedFormulas,
            logFormulasKey: self.facts,
            categoricalsKey: self.categoricalConstraints,
            evidenceKey: self.evidence
        }, savePath)

    def include(self, secondHybridKB):
        self.weightedFormulas = {**self.weightedFormulas,
                                 **secondHybridKB.weightedFormulas}
        self.facts = {**self.facts,
                      **secondHybridKB.facts}
        self.categoricalConstraints = {**self.categoricalConstraints,
                                       **secondHybridKB.categoricalConstraints}
        self.evidence = {**self.evidence,
                         **secondHybridKB.evidence}
        self.find_atoms()

    def create_cores(self):
        return {**encoding.create_formulas_cores({**self.weightedFormulas, **self.facts}),
                **encoding.create_evidence_cores(self.evidence),
                **encoding.create_categorical_cores(self.categoricalConstraints),
                **encoding.create_atomization_cores([atom for atom in self.atoms if "=" in atom], self.dimensionDict),
                **self.backCores}

    def get_partition_function(self, allAtoms=[]):
        unseenAtomNum = len([atom for atom in allAtoms if atom not in self.atoms])
        return (engine.contract(coreDict=self.create_cores(), openColors=[]).values
                * (2 ** unseenAtomNum))

    def create_hard_cores(self):
        """
        Returns the cores posing hard logical constraints on the worlds to be models
        """
        return {**encoding.create_formulas_cores(self.facts),
                **encoding.create_evidence_cores(self.evidence),
                **encoding.create_categorical_cores(self.categoricalConstraints),
                **self.backCores}

    def is_satisfiable(self):
        """
        Decides whether the Knowledge Base is satisfiable, i.e. whether a model exists
        """
        return engine.contract(coreDict=self.create_hard_cores(),
                               openColors=[]).values > 0
