from flask import Flask, request, jsonify
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain.agents import Tool
import os
from dotenv import load_dotenv
from typing import Annotated, List
import operator
from typing_extensions import TypedDict
from flask_cors import CORS


load_dotenv()

app = Flask(__name__)
CORS(app, origins=["*"])

# Initialize LangGraph components
memory = MemorySaver()


class State(TypedDict):
    # Use operator.add for list concatenation
    messages: Annotated[List[str], operator.add]


graph_builder = StateGraph(State)

llm = ChatOpenAI(
    temperature=0.5, model="gpt-3.5-turbo", api_key=os.getenv("OPENAI_API_KEY")
)

def generate_joke(input: str) -> str:
    prompt_template = """
    Create a unique joke. The joke should be in the format:
    Setup: [Setup of the joke]
    Punchline: [Punchline of the joke]
    """
    prompt = PromptTemplate(input_variables=[], template=prompt_template)
    runnable_sequence = RunnableSequence(prompt | llm)
    result = runnable_sequence.invoke({})
    joke = result if isinstance(result, str) else result.content
    return joke.strip()


def rewrite_joke(input: str, *args, **kwargs) -> str:
    prompt_template = f"""
    write a joke that is completely different from the previous joke:
    it should be about the tags:
    """
    prompt = PromptTemplate(input_variables=[], template=prompt_template)
    runnable_sequence = RunnableSequence(prompt | llm)
    result = runnable_sequence.invoke({})
    new_joke = result if isinstance(result, str) else result.content
    return new_joke.strip()

Generate_Joke_Tool = Tool(
    name="GenerateJoke", func=generate_joke, description="Use this tool when user ask to tell a joke")
Rewrite_Joke_Tool = Tool(name="RewriteJoke", func=rewrite_joke,
                         description="Rewrite a new joke based on user feedback")
tools = [Generate_Joke_Tool, Rewrite_Joke_Tool]
llm_with_tools = llm.bind_tools(tools)


def human_feedback(state):
    # Logic to handle human feedback can be added here
    # For now, just pass the messages through
    message = input("Enter your message: ")
    return {"messages": ["haha, that was funny!"]}


graph_builder.add_node("human_feedback", human_feedback)
graph_builder.add_node("chatbot", lambda state: {"messages": [
                       llm_with_tools.invoke(state["messages"])]})
tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph_builder.add_conditional_edges("chatbot", tools_condition)
graph_builder.add_edge("tools", "human_feedback")
graph_builder.add_edge("human_feedback", "chatbot")

graph = graph_builder.compile(checkpointer=memory, interrupt_after=["human_feedback"])


@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('input', '')
    state = {"messages": [user_input]}
    # Ensure 'thread_id' is provided
    config = {"configurable": {"thread_id": "1"}}
    events = graph.stream(state, config, stream_mode="values")

    # Convert messages to strings
    responses = []
    for event in events:
        print(event)
        if "messages" in event:
            # Convert each message to a string
            message_content = str(event["messages"][-1])
            responses.append(message_content)

    snapshot = graph.get_state(config)
    print(snapshot.next)

    return jsonify(responses)


if __name__ == '__main__':
    app.run(port=8080)
