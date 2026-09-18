def chatbot(question):
    if question == "What is your name?":
        return "I am a xyz chatbot"
    elif question == "What can you do?":
        return "I can answer basic questions"
    else:
        return "I don't know"


# Test 1
question1 = "What is your name?"
actual1 = chatbot(question1)
expected1 = "I am a xyz chatbot"

if actual1 == expected1:
    print("Test 1: PASS")
else:
    print("Test 1: FAIL, got:", actual1)


# Test 2
question2 = "What can you do?"
actual2 = chatbot(question2)
expected2 = "I can answer basic questions"

if actual2 == expected2:
    print("Test 2: PASS")
else:
    print("Test 2: FAIL, got:", actual2)


# Test 3
question3 = "Random question"
actual3 = chatbot(question3)
expected3 = "I don't know"

if actual3 == expected3:
    print("Test 3: PASS")
else:
    print("Test 3: FAIL, got:", actual3)