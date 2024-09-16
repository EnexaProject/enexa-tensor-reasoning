from tnreason import engine


class KikuchiContractor:
    def __init__(self, colorDict, coreDict):
        """
        colorDict: colors to each hyperedge
        coreDict: cores to each hyperedge (can be empty)
        """
        self.colorDict = colorDict
        self.coreDict = coreDict
        self.directedEdges = {parent: [child for child in self.colorDict if
                                       set(self.colorDict[child]) <= set(self.colorDict[parent]) and child != parent]
                              for parent in self.colorDict}
        self.messages = {parent: {child: {} for child in self.directedEdges[parent]} for parent in self.colorDict}

    def update_message(self, parent, child):
        coreDict = self.coreDict[parent]
        for otherChild in self.directedEdges[parent]:
            if otherChild != child:
                coreDict.update(self.coreDict[child])
                for otherParent in self.directedEdges:
                    if otherParent != parent and otherParent not in self.directedEdges[parent] and otherChild in \
                            self.directedEdges[otherParent]:
                        print("otherParent", otherParent, "otherChild", otherChild)
                        coreDict.update(self.messages[otherParent][otherChild])
        print(coreDict)
        self.messages[parent][child] = {
            parent + "_" + child + "_mesCore": engine.contract(coreDict, openColors=self.colorDict[child])}


if __name__ == "__main__":
    kiki = KikuchiContractor(colorDict={"e1": ["a", "b"], "v1": ["a"], "v2": ["b"], "e2": ["a", "b", "c"]},
                             coreDict={"e1": {}, "v1": {}, "v2": {}, "e2": {}
                                       })
    # kiki.find_poset_diagram()
    print(kiki.directedEdges)
    print(kiki.messages)

    kiki.update_message("e1", "v1")
