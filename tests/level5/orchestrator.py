import ollama
from specialists import run_math_agent, run_time_agent


# The meta agent doesn't do the task itself.
# It only decides WHICH specialist agent should handle the question.

routing_tools = [
    {
        "type": "function",
        "function": {
            "name": "math_specialist",
            "description": "Handles arithmetic questions like addition or multiplication",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "time_specialist",
            "description": "Handles questions about the current date or time",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


def run_orchestrator(question):

    messages = [{"role": "user", "content": question}]

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=messages,
        tools=routing_tools
    )

    message = response["message"]

    if message.get("tool_calls"):
        chosen_specialist = message["tool_calls"][0]["function"]["name"]

        if chosen_specialist == "math_specialist":
            answer, tools_used = run_math_agent(question)
            return answer, "math_specialist", tools_used

        elif chosen_specialist == "time_specialist":
            answer, tools_used = run_time_agent(question)
            return answer, "time_specialist", tools_used

    # No specialist matched — meta agent answers directly
    return message["content"], None, []


if __name__ == "__main__":
    question = "What is 25 times 4?"
    answer, specialist_used, tools_used = run_orchestrator(question)

    print("Specialist used:", specialist_used)
    print("Tools used:", tools_used)
    print("Answer:", answer)