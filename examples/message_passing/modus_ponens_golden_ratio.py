from examples.message_passing import mp_moments as mpm

expressionsDict = {
    "e1": ["imp", "p", "q"],
}

empiricalMeanDict = {
    "e1": 1,
}

matcher = mpm.MPMomentMatcher(expressionsDict=expressionsDict, empiricalMeanDict=empiricalMeanDict)

matcher.upward_messages()
matcher.adjust_headCores()

for i in range(100):
    matcher.upward_messages()
    matcher.adjust_headCores()
    matcher.downward_messages()

for key in matcher.messageCores:
    print(key, matcher.messageCores[key].values)

