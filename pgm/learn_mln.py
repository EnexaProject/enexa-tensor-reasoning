import expression_learning as el
class  PGMLearner:
    def __init__(self,skeletonDict,candidatesSuperDict):
        self.skeletonDict = skeletonDict
        self.candidatesSuperDict = candidatesSuperDict
        self.solutionDict = {}
    def learn_expression(self,skeletonKey):
        learner = el.ExpressionLearner(skeletonExpression=skeletonDict[skeletonKey])

    def learn_weight(self,skeletonKey):
        pass

