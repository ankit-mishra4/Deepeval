from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from agentflow import generate_plan


# Judge model that will read the plan and decide if it is a good plan.
model = OllamaModel(
    model="qwen2.5:7b",
    base_url="http://localhost:11434"
)


def test_plan_quality():

    question = "First add 10 and 5, then multiply that result by 3."

    # Ask the agent to write its plan in plain text, before it does anything.
    plan = generate_plan(question)

    # The test case just holds the question and the plan text.
    test_case = LLMTestCase(
        input=question,
        actual_output=plan
    )

    # GEval lets us write our own custom judging criteria in plain English.
    # Here we are checking: is this plan logical and complete for the task?
    metric = GEval(
        name="PlanQuality",
        criteria=(
            "Determine if the plan in the actual output is logical, complete, "
            "and would correctly solve the task described in the input, using "
            "only the tools add, multiply, and get_current_time."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT
        ],
        threshold=0.5,
        model=model
    )

    metric.measure(test_case)

    print("Question:", question)
    print("Generated plan:", plan)
    print("Plan Quality score:", metric.score)
    print("Reason:", metric.reason)

    assert metric.score >= 0.5