import pandas as pd

from tnreason.model import create_mln as cmln
from tnreason.representation import sampledf_to_factdf as stof, sampledf_to_pairdf as stop

from pgmpy.sampling import GibbsSampling

def get_last_as_dict(sampler, chainSize):
    return sampler.sample(size=chainSize).iloc[-1].to_dict()

def generate_samples(model, sampleNum, chainSize):
    df = pd.DataFrame(columns=model.nodes)
    for index in range(sampleNum):
        sampler = GibbsSampling(model)
        row_df = pd.DataFrame(get_last_as_dict(sampler,chainSize),index=[index])
        df = pd.concat([df, row_df])
    return df

def generate_factDf_and_pairDf(expressionDict,sampleNum,chainSize = 10, prefix ="tev"):
    mln = cmln.create_markov_logic_network(expressionDict)
    sampleDf = generate_samples(mln,sampleNum = sampleNum, chainSize=chainSize)
    factDf = stof.sampleDf_to_factDf(sampleDf, prefix = prefix)
    pairDf = stop.sampleDf_to_pairDf(sampleDf, prefix = prefix)
    return factDf, pairDf