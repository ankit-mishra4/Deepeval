from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel
from my_agent import ask_agent


model = OllamaModel(
    model="qwen2.5:7b",
    base_url="http://localhost:11434"
)


def test_calculator_tool():

    question = "What is 25 times 4?"
    answer, tool_used = ask_agent(question)

    print("Tool used:", tool_used)
    print("Answer:", answer)

    assert tool_used == "calculator"


def test_datetime_tool():

    question = "What is the current time?"
    answer, tool_used = ask_agent(question)

    print("Tool used:", tool_used)
    print("Answer:", answer)

    assert tool_used == "get_current_datetime"


def test_no_tool_needed():

    question = "What is the capital of France?"
    answer, tool_used = ask_agent(question)

    print("Tool used:", tool_used)
    print("Answer:", answer)

    assert tool_used is None


def test_answer_relevancy():

    question = "What is 25 times 4?"
    answer, tool_used = ask_agent(question)

    test = LLMTestCase(
        input=question,
        actual_output=answer
    )

    metric = AnswerRelevancyMetric(threshold=0.5, model=model)
    metric.measure(test)

    print("Score:", metric.score)

    assert metric.score >= 0.5