from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval
from deepeval.models import OllamaModel
from agentflow import run_agentflow


model = OllamaModel(
    model="qwen2.5:7b",
    base_url="http://localhost:11434"
)


def test_step_efficiency():

    question = "First add 10 and 5, then multiply that result by 3."

    # Run the real agent and see exactly which tools it used.
    answer, tools_used, trace = run_agentflow(question)

    execution_summary = "Tools executed in this order: " + ", ".join(tools_used)

    test_case = LLMTestCase(
        input=question,
        actual_output=execution_summary
    )

    # This checks that the agent did not repeat a tool unnecessarily,
    # and did not take more steps than the task actually required.
    metric = GEval(
        name="StepEfficiency",
        criteria=(
            "Determine if the sequence of tool calls in the actual output "
            "was efficient for solving the task in the input — meaning no "
            "tool was called more times than necessary, and no unnecessary "
            "or redundant steps were taken."
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
    print("Execution:", execution_summary)
    print("Step Efficiency score:", metric.score)
    print("Reason:", metric.reason)

    assert metric.score >= 0.5


def test_step_efficiency_no_duplicate_calls():

    # A simpler, non-LLM check: for this specific task, each tool
    # should only need to be called once. If a tool appears more
    # than once, that is a sign of a wasted or repeated step.

    question = "First add 10 and 5, then multiply that result by 3."
    answer, tools_used, trace = run_agentflow(question)

    print("Tools used:", tools_used)

    for tool_name in set(tools_used):
        count = tools_used.count(tool_name)
        assert count == 1, f"{tool_name} was called {count} times, expected only 1"