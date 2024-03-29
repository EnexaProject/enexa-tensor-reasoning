
from tnreason.learning import formula_sampling as fs


skeleton = ["P0", "con1", "P1"]
candidatesDict = {
        "P0": ["sikorka", "sledz"],
        "P1": ["jaszczur"],
        "con1" : ["and","or","imp"]
    #], "piskle"]#, "szczeniak", "piskle"]
    }

from tnreason.model import generate_test_data as gtd
sampleDf = gtd.generate_sampleDf({
        "f1": [["sikorka", "and", ["not","piskle"]], 2],
        "f2": [[["not","sledz"], "and", ["not","szczeniak"]], 20],
        "f3": [["jaszczur", "and", "sikorka"], 2],
    }, 10)

fSampler = fs.GibbsFormulaSampler(skeleton, candidatesDict, sampleDf=sampleDf)
print("likelihood",fSampler.compute_loglikelihood(["not","sikorka"],1))

fSampler.gibbs(10)
print(fSampler.assignment)



fSampler.gibbs_simulated_annealing([(10,1),(10,0.5),(10,0.2),(10,0.1)])
print(fSampler.assignment)