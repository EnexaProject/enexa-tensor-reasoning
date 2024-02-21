import tnreason.model.generate_test_data as gtd
import tnreason.tensor.formula_tensors as ft

import time

sampleNum = 100
atomNum = 5
testDict = {
    "t"+str(i) : ["a"+str(i), 2] for i in range(atomNum)
}

sampleDf = gtd.generate_sampleDf(testDict, sampleNum)
copyNum = 4
for k in range(copyNum):
    for i in range(atomNum):
        sampleDf["a"+str(i)+"_"+str(k)] = sampleDf["a"+str(i)]


dataTensor = ft.DataTensor(sampleDf)

starttime = time.time()
print("Entropy Start")
print("Entropy with PGMPY",dataTensor.compute_shannon_entropy(contractionMethod="PgmpyVariableEliminator"))
endtime = time.time()
print(endtime - starttime)
print("Entropy with TNChainContractor",dataTensor.compute_shannon_entropy(contractionMethod="TNChainContractor"))
endtime2 =time.time()
print(endtime2-endtime)
