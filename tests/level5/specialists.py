
import ollama
from toolkit import add, multiply, get_current_time



# MATH SPECIALIST — only knows about add and multiply


math_tools = [
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
    }
]


def run_math_agent(question, max_steps=5):

    messages = [{"role": "user", "content": question}]
    tools_used = []

    for step in range(max_steps):
        response = ollama.chat(model="qwen2.5:7b", messages=messages, tools=math_tools)
        message = response["message"]

        if message.get("tool_calls"):
            call = message["tool_calls"][0]
            tool_name = call["function"]["name"]
            tool_args = call["function"]["arguments"]

            if tool_name == "add":
                tool_result = str(add(tool_args["a"], tool_args["b"]))
            elif tool_name == "multiply":
                tool_result = str(multiply(tool_args["a"], tool_args["b"]))
            else:
                tool_result = "Unknown tool"

            tools_used.append(tool_name)
            messages.append(message)
            messages.append({"role": "tool", "content": tool_result})
        else:
            return message["content"], tools_used

    return "Math agent stopped after max steps.", tools_used



# TIME SPECIALIST — only knows about get_current_time


time_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    }
]


def run_time_agent(question, max_steps=3):

    messages = [{"role": "user", "content": question}]
    tools_used = []

    for step in range(max_steps):
        response = ollama.chat(model="qwen2.5:7b", messages=messages, tools=time_tools)
        message = response["message"]

        if message.get("tool_calls"):
            call = message["tool_calls"][0]
            tool_name = call["function"]["name"]

            if tool_name == "get_current_time":
                tool_result = str(get_current_time())
            else:
                tool_result = "Unknown tool"

            tools_used.append(tool_name)
            messages.append(message)
            messages.append({"role": "tool", "content": tool_result})
        else:
            return message["content"], tools_used

    return "Time agent stopped after max steps.", tools_used