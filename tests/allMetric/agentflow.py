import ollama
from toolkit import add, multiply, get_current_time


tools = [
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "Add two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "multiply",
            "description": "Multiply two numbers",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


def run_tool(tool_name, tool_args):
    if tool_name == "add":
        return str(add(tool_args["a"], tool_args["b"]))
    elif tool_name == "multiply":
        return str(multiply(tool_args["a"], tool_args["b"]))
    elif tool_name == "get_current_time":
        return str(get_current_time())
    else:
        return "Unknown tool"


def generate_plan(question):
    # This asks the model to write down its plan in plain text,
    # BEFORE it actually starts calling any tools.
    # This plan is only used for evaluation later — it does not
    # control what the agent actually does.

    prompt = (
        "Write a short step-by-step plan for how you would answer this "
        "question using the available tools: add, multiply, get_current_time. "
        "Just list the steps in plain text, do not solve it yet.\n\n"
        f"Question: {question}"
    )

    response = ollama.chat(
        model="qwen2.5:7b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def run_agentflow(question, max_steps=5):

    messages = [{"role": "user", "content": question}]

    tools_used = []
    trace = []

    for step in range(max_steps):

        response = ollama.chat(
            model="qwen2.5:7b",
            messages=messages,
            tools=tools
        )

        message = response["message"]

        if message.get("tool_calls"):

            call = message["tool_calls"][0]
            tool_name = call["function"]["name"]
            tool_args = call["function"]["arguments"]

            tool_result = run_tool(tool_name, tool_args)
            tools_used.append(tool_name)

            trace.append({
                "name": tool_name,
                "input_parameters": tool_args,
                "output": tool_result
            })

            messages.append(message)
            messages.append({"role": "tool", "content": tool_result})

        else:
            final_answer = message["content"]
            return final_answer, tools_used, trace

    final_answer = "Stopped after max steps without finishing."
    return final_answer, tools_used, trace


if __name__ == "__main__":
    question = "First add 10 and 5, then multiply that result by 3."

    plan = generate_plan(question)
    print("Generated plan:\n", plan)

    answer, tools_used, trace = run_agentflow(question)
    print("\nTools used (in order):", tools_used)
    print("Answer:", answer)