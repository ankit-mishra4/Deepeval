from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.models import OllamaModel
from agentflow import run_agentflow


# This is the judge model. It reads the question and the arguments
# that were actually passed to the tool, and decides if those
# arguments were extracted correctly.
model = OllamaModel(
    model="qwen2.5:7b",
    base_url="http://localhost:11434"
)


def build_tool_calls(trace):
    # Converts our own trace (a list of plain dictionaries) into the
    # ToolCall objects that DeepEval expects.
    tool_calls = []

    for step in trace:
        tool_calls.append(
            ToolCall(
                name=step["name"],
                input_parameters=step["input_parameters"],
                output=step["output"]
            )
        )

    return tool_calls


def test_argument_correctness():

    question = "What is 25 times 4?"

    # Run the real agent and get its actual execution trace.
    answer, tools_used, trace = run_agentflow(question)

    tool_calls = build_tool_calls(trace)

    # Build the test case: the original question, the final answer,
    # and the tools that were actually called with their arguments.
    test_case = LLMTestCase(
        input=question,
        actual_output=answer,
        tools_called=tool_calls
    )

    # This metric checks: were the values passed into the tool correct,
    # not just whether the right tool was picked.
    metric = ArgumentCorrectnessMetric(threshold=0.5, model=model)
    metric.measure(test_case)

    print("Question:", question)
    print("Tool calls:", tool_calls)
    print("Argument Correctness score:", metric.score)
    print("Reason:", metric.reason)

    assert metric.score >= 0.5