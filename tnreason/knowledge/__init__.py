# knowledge/__init__.py

from tnreason.knowledge.inductive import HybridLearner
from tnreason.knowledge.deductive import InferenceProvider

from tnreason.knowledge.weight_estimation import EntropyMaximizer
from tnreason.knowledge.distributions import HybridKnowledgeBase, EmpiricalDistribution
from tnreason.knowledge.formula_boosting import FormulaBooster
from tnreason.knowledge.batch_evaluation import KnowledgePropagator

from tnreason.knowledge.knowledge_visualization import visualize

def load_kb_from_yaml(loadPath):
    kb = HybridKnowledgeBase()
    kb.from_yaml(loadPath)
    return kb


