#from tnreason.knowledge import tensor_kb as tkb

from tnreason import knowledge as know

if __name__ == "__main__":
    tensorKB = know.HardKnowledgeBase(["a3", ["not", [["not", "a1"], "and", "a2"]]])
    assert tensorKB.ask("a1") == "contingent"

    tensorKB.tell("a2")
    tensorKB.tell(["not","a2"])
    tensorKB.tell("a2")
    assert tensorKB.ask("a1") == "entailed"

    assert tensorKB.ask("a4") == "contingent"

    tensorKB.tell(["not", "a4"])
    assert tensorKB.ask("a4") == "contradicting"