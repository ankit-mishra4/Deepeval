from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from agentflow import generate_plan, run_agentflow


model = OllamaModel(
    model="qwen2.5:7b",
    base_url="http://localhost:11434"
)


def test_plan_adherence():

    question = "First add 10 and 5, then multiply that result by 3."

    # Step 1: get the plan the agent said it would follow.
    plan = generate_plan(question)

    # Step 2: actually run the agent and see what it really did.
    answer, tools_used, trace = run_agentflow(question)

    # Turn the actual execution into a simple readable text,
    # so the judge model can compare it against the plan.
    actual_execution = "Tools executed in this order: " + ", ".join(tools_used)

    # The plan goes in expected_output, and what actually happened
    # goes in actual_output. This is just how we are choosing to
    # store the two things being compared — GEval does not care
    # about the field names, only what criteria we give it.
    test_case = LLMTestCase(
        input=question,
        actual_output=actual_execution,
        expected_output=plan
    )

    metric = GEval(
        name="PlanAdherence",
        criteria=(
            "Determine if the actual output (the tools that were really "
            "executed) follows the plan described in the expected output. "
            "The execution does not need to match word for word, but the "
            "same steps, in the same order, should have been carried out."
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        threshold=0.5,
        model=model
    )

    metric.measure(test_case)

    print("Question:", question)
    print("Plan:", plan)
    print("Actual execution:", actual_execution)
    print("Plan Adherence score:", metric.score)
    print("Reason:", metric.reason)

    assert metric.score >= 0.5