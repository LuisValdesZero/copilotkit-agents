"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

from typing import Any, List
import json
from typing_extensions import Literal
import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode
from copilotcloud_rl import getLearningContext

class AgentState(MessagesState):
    """
    Here we define the state of the agent

    In this instance, we're inheriting from CopilotKitState, which will bring in
    the CopilotKitState fields. We're also adding a custom field, `language`,
    which will be used to set the language of the agent.
    """
    proverbs: List[str] = []
    tools: List[Any]
    name: str = ""

@tool
def get_weather(location: str):
    """
    Get the weather for a given location.
    """
    return f"The weather for {location} is 70 degrees."

@tool
def set_name(name: str):
    """
    Update the agent's name. Always call this when the user asks to change or set the name.
    """
    return json.dumps({"action": "set_name", "name": name})

@tool
def generate_proverb(topic: str = ""):
    """
    Generate a concise, wise proverb using the LLM and persist it into the agent state.
    Call this when the user asks to generate a proverb. Optionally pass a short 'topic'.
    """
    model = ChatOpenAI(model="gpt-4o")
    prompt = (
        "Create a single, concise, original proverb. \n"
        "- Style: universal wisdom, timeless, <= 18 words.\n"
        "- No quotes, no explanations, just the proverb.\n"
        f"- Topic (optional): {topic.strip()}\n"
    )
    try:
        result = model.invoke([SystemMessage(content="You coin crisp, wise proverbs."), HumanMessage(content=prompt)])
        text = result.content if isinstance(result.content, str) else str(result.content)
        proverb = text.strip().strip('"')
    except Exception:
        proverb = "Wisdom whispers where patience walks."

    return json.dumps({"action": "add_proverb", "proverb": proverb})

# @tool
# def your_tool_here(your_arg: str):
#     """Your tool description here."""
#     print(f"Your tool logic here")
#     return "Your tool response here."

backend_tools = [
    get_weather,
    set_name,
    generate_proverb
    # your_tool_here
]

# Extract tool names from backend_tools for comparison
backend_tool_names = [tool.name for tool in backend_tools]


async def chat_node(state: AgentState, config: RunnableConfig) -> Command[Literal["tool_node", "__end__"]]:
    """
    Standard chat node based on the ReAct design pattern. It handles:
    - The model to use (and binds in CopilotKit actions and the tools defined above)
    - The system prompt
    - Getting a response from the model
    - Handling tool calls

    For more about the ReAct design pattern, see:
    https://www.perplexity.ai/search/react-agents-NcXLQhreS0WDzpVaS4m9Cg
    """

    # 0. Tool execution and state updates are handled in the dedicated tool node

    # 1. Define the model
    model = ChatOpenAI(model="gpt-4o")

    # 2. Bind the tools to the model
    model_with_tools = model.bind_tools(
        [
            *state.get("tools", []), # bind tools defined by ag-ui
            *backend_tools,
            # your_tool_here
        ],

        # 2.1 Disable parallel tool calls to avoid race conditions,
        #     enable this for faster performance if you want to manage
        #     the complexity of running tool calls in parallel.
        parallel_tool_calls=False,
    )

    # 3. Define the system message by which the chat model will be run
    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. "
            f"The current proverbs are {state.get('proverbs', [])}. "
            "When the user asks to change or set the agent's name, call the set_name tool with the requested name. "
            "When the user asks to generate a proverb, call generate_proverb with a short topic summarizing their request."
        )
    )

    messages = state.get("messages", [])
    # print(f"#################### messages: {messages}")


    # To the last HumanMessage add the learning context in the prompt
    updated_messages = list(state.get("messages", []))
    # find last HumanMessage
    # Find the last message and check if it is a HumanMessage
    if updated_messages and isinstance(updated_messages[-1], HumanMessage):

        human_message = updated_messages[-1]
        prompt = human_message.content

        print(f"#################### prompt: {prompt}")
        try:
            result = await asyncio.to_thread(
                getLearningContext, prompt=prompt, agentName="sample_agent"
            )
            learningContext = result["learningContext"]
            print(f"#################### learning_context: {learningContext}")
            learning_context_text = learningContext if isinstance(learningContext, str) else json.dumps(learningContext)
            injected = f"{prompt}\n\n[Learning context]\n{learning_context_text}"
            updated_messages[-1] = HumanMessage(content=injected, additional_kwargs=getattr(human_message, "additional_kwargs", {}), response_metadata=getattr(human_message, "response_metadata", {}))
            state["messages"] = updated_messages
        except Exception as error:
            print(f"#################### getLearningContext error: {error}")

    # 4. Run the model to generate a response
    response = await model_with_tools.ainvoke([
        system_message,
        *state["messages"],
    ], config)

    # only route to tool node if tool is not in the tools list
    if route_to_tool_node(response):
        print("routing to tool node")
        return Command(
            goto="tool_node",
            update={
                "messages": [response],
            }
        )

    # 5. We've handled all tool calls, so we can end the graph.
    return Command(
        goto=END,
        update={
            "messages": [response],
        }
    )

def route_to_tool_node(response: BaseMessage):
    """
    Route to tool node if any tool call in the response matches a backend tool name.
    """
    tool_calls = getattr(response, "tool_calls", None)
    if not tool_calls:
        return False

    for tool_call in tool_calls:
        if tool_call.get("name") in backend_tool_names:
            return True
    return False

def _extract_latest_tool_calls(messages: List[Any]):
    for msg in reversed(messages or []):
        if hasattr(msg, "tool_calls") and getattr(msg, "tool_calls"):
            return getattr(msg, "tool_calls")
    return []


async def tool_node(state: AgentState, config: RunnableConfig):
    updates: dict = {}
    new_messages: List[Any] = []

    tool_calls = _extract_latest_tool_calls(state.get("messages", []))
    if not tool_calls:
        # Nothing to do, go back to chat
        return Command(goto="chat_node", update={})

    for tc in tool_calls:
        name = tc.get("name")
        args = tc.get("args", {}) or {}
        tc_id = tc.get("id")

        if name == "set_name":
            new_name = str(args.get("name", "")).strip()
            if new_name:
                updates["name"] = new_name
                new_messages.append(ToolMessage(content=json.dumps({"status": "ok", "name": new_name}), tool_call_id=tc_id))

        elif name == "generate_proverb":
            topic = str(args.get("topic", "")).strip()
            model = ChatOpenAI(model="gpt-4o")
            prompt = (
                "Create a single, concise, original proverb. \n"
                "- Style: universal wisdom, timeless, <= 18 words.\n"
                "- No quotes, no explanations, just the proverb.\n"
                f"- Topic (optional): {topic}\n"
            )
            try:
                result = await model.ainvoke([SystemMessage(content="You coin crisp, wise proverbs."), HumanMessage(content=prompt)], config)
                text = result.content if isinstance(result.content, str) else str(result.content)
                proverb_text = text.strip().strip('"')
            except Exception:
                proverb_text = "Wisdom whispers where patience walks."

            current = state.get("proverbs", []) or []
            updates["proverbs"] = [*current, proverb_text]
            new_messages.append(ToolMessage(content=json.dumps({"status": "ok", "proverb": proverb_text}), tool_call_id=tc_id))

    if new_messages:
        updates["messages"] = new_messages

    return Command(goto="chat_node", update=updates)


# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", tool_node)
workflow.add_edge("tool_node", "chat_node")
workflow.set_entry_point("chat_node")

graph = workflow.compile()
