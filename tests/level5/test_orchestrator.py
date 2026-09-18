from orchestrator import run_orchestrator


def test_math_question_routes_to_math_specialist():
    question = "What is 25 times 4?"
    answer, specialist_used, tools_used = run_orchestrator(question)

    print("Specialist used:", specialist_used)
    print("Tools used:", tools_used)
    print("Answer:", answer)

    assert specialist_used == "math_specialist"


def test_time_question_routes_to_time_specialist():
    question = "What is the current time?"
    answer, specialist_used, tools_used = run_orchestrator(question)

    print("Specialist used:", specialist_used)
    print("Tools used:", tools_used)
    print("Answer:", answer)

    assert specialist_used == "time_specialist"


def test_general_question_uses_no_specialist():
    question = "What is the capital of France?"
    answer, specialist_used, tools_used = run_orchestrator(question)

    print("Specialist used:", specialist_used)
    print("Answer:", answer)

    assert specialist_used is None


# CONVERSATIONAL AI TEST


def test_conversational_ai():
    conversation = []

    # First question
    question1 = "What is 25 times 4?"

    answer1, specialist1, tools1 = run_orchestrator(
        question1,
        conversation
    )

    print("Question 1:", question1)
    print("Answer 1:", answer1)
    print("Specialist 1:", specialist1)

    assert specialist1 == "math_specialist"
    assert "100" in answer1

    # Second question - depends on previous conversation
    question2 = "Now add 50 to that."

    answer2, specialist2, tools2 = run_orchestrator(
        question2,
        conversation
    )

    print("Question 2:", question2)
    print("Answer 2:", answer2)
    print("Specialist 2:", specialist2)

    assert specialist2 == "math_specialist"
    assert "150" in answer2