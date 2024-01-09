from tnreason.model import sampling as sam
from tnreason.representation import sampledf_to_factdf as stof, sampledf_to_pairdf as stop

def generate_sampleDf(expressionDict, sampleNum, method="Gibbs10"):
    if method.startswith("Gibbs"):
        return sam.GibbsSampler(expressionDict).create_sampleDf(sampleNum, int(method.split("Gibbs")[1]))

def generate_factDf_and_pairDf(expressionDict,sampleNum,chainSize = 10, prefix ="tev"):
    sampleDf = generate_sampleDf(expressionDict, sampleNum, chainSize=chainSize)
    factDf = stof.sampleDf_to_factDf(sampleDf, prefix = prefix)
    pairDf = stop.sampleDf_to_pairDf(sampleDf, prefix = prefix)
    return factDf, pairDf

