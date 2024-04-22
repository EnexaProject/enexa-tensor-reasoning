from tnreason import knowledge

from tnreason.algorithms import constraint_propagation as cp

import numpy as np

hybridKB = knowledge.HybridKnowledgeBase(
    {},  # "f1":[["a1","and",["not","a3"]], 10]},
    facts={"funfact" : ["a6", "or", "a7"],
                "fact1": ["a1", "xor", "a2"],
               "fact2" : "a4",
                "fact3" : "a1"},
    categoricalConstraints={
        "c1": ["a1", "a2", "a3"],
        "c2": ["a4"]
    }
)

print(hybridKB.facts.get_cores().keys())

propagator = cp.ConstraintPropagator(
    {**hybridKB.facts.get_cores(),
     **hybridKB.formulaTensors.get_cores()})

propagator.propagate_cores()
evidenceDict, multipleAssignments, redCores, remCores = propagator.find_evidence_and_redundant_cores()

print(remCores)

#from tnreason.contraction import contraction_visualization as cv
#cv.draw_contractionDiagram(propagator.binaryCoresDict)