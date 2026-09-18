from agentflow import run_agentflow


def test_single_tool_still_works():
    question = "What is 25 times 4?"
    answer, tools_used = run_agentflow(question)

    print("Tools used:", tools_used)
    print("Answer:", answer)

    assert "multiply" in tools_used


def test_multi_step_planning():
    question = "First add 10 and 5, then multiply that result by 3."
    answer, tools_used = run_agentflow(question)

    print("Tools used:", tools_used)
    print("Answer:", answer)

    assert "add" in tools_used
    assert "multiply" in tools_used
    assert tools_used.index("add") < tools_used.index("multiply")


def test_no_tool_needed():
    question = "What is the capital of France?"
    answer, tools_used = run_agentflow(question)

    print("Tools used:", tools_used)
    print("Answer:", answer)

    assert tools_used == []