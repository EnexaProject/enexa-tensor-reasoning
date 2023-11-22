from tnreason.model import markov_logic_network as mln
from tnreason.representation import sampledf_to_factdf as stof, sampledf_to_pairdf as stop

def generate_sampleDf(expressionDict, sampleNum, chainSize=10):
    markovNetwork = mln.MarkovLogicNetwork(expressionDict)
    return markovNetwork.generate_sampleDf(sampleNum,chainSize)

def generate_factDf_and_pairDf(expressionDict,sampleNum,chainSize = 10, prefix ="tev"):
    sampleDf = generate_sampleDf(expressionDict, sampleNum, chainSize=chainSize)
    factDf = stof.sampleDf_to_factDf(sampleDf, prefix = prefix)
    pairDf = stop.sampleDf_to_pairDf(sampleDf, prefix = prefix)
    return factDf, pairDf

