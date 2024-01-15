from tnreason.model import tensor_kb as tkb

if __name__ == "__main__":
    tensorKB = tkb.TensorKB(["a3", ["not", [["not", "a1"], "and", "a2"]]])
    assert tensorKB.ask("a1") == "contingent"

    tensorKB.tell("a2")
    tensorKB.tell(["not","a2"])
    tensorKB.tell("a2")
    assert tensorKB.ask("a1") == "entailed"

    assert tensorKB.ask("a4") == "contingent"

    tensorKB.tell(["not", "a4"])
    assert tensorKB.ask("a4") == "contradicting"