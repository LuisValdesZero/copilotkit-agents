"""
This is the main entry point for the agent.
It defines the workflow graph, state, tools, nodes and edges.
"""

import json
import asyncio
from typing import Any, List, Annotated, Optional
from pydantic import Field
from typing_extensions import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain.tools import tool
from langchain_core.tools import InjectedToolCallId
from langgraph.graph import StateGraph, END
from langgraph.types import Command
from langgraph.graph import MessagesState
from langgraph.prebuilt import ToolNode, InjectedState
from copilotcloud_rl import getLearningContext

class AgentState(MessagesState):
    """
    Defines the state of the agent.

    This class extends MessagesState to include additional fields for agent-specific data.
    For example, you can add custom fields such as `proverbs` (a list of generated proverbs)
    or `name` (the agent's current name) to persist information across the agent's workflow.
    """
    proverbs: Optional[List[str]] = Field(default_factory=list)
    name: str = ""

@tool
def get_weather(location: str):
    """
    Get the weather for a given location.
    """
    return f"The weather for {location} is 70 degrees."

@tool
def set_name(name: str, tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Sets or updates the agent's name.

    Parameters:
        name (str): The new name to assign to the agent.
        tool_call_id (str): The unique identifier for this tool call (injected automatically).

    Usage:
        Use this tool whenever the user requests to set or change the agent's name.
    """
    return Command(update={
        "name": name,
        "messages": [
            ToolMessage(f"Updated user name to {name}", tool_call_id=tool_call_id)
        ]
    })

@tool
def generate_proverb(topic: str, state: Annotated[dict, InjectedState], tool_call_id: Annotated[str, InjectedToolCallId]) -> Command:
    """
    Generates a concise, original proverb using a language model and saves it to the agent's state.

    This tool should be called when the user requests a proverb. You may optionally provide a short 'topic'
    to guide the proverb's theme.

    The generated proverb is:
    - Universal in wisdom and timeless in style
    - No longer than 18 words
    - Returned without quotes or explanations

    The new proverb is appended to the agent's list of proverbs and a message is added to the state.

    Parameters:
        topic (str): An optional topic to inspire the proverb.
        state (AgentState): The current state of the agent (injected automatically).
        tool_call_id (str): The unique identifier for this tool call (injected automatically).

    Returns:
        Command: An update command containing the new proverb and a message.
    """
    
    if 'proverbs' not in state:
        state['proverbs'] = []
    current_proverbs = state.get('proverbs', [])

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
    except Exception as error:
        print(f"#################### error: {error}")
        proverb = "Wisdom whispers where patience walks."

    proverbs = current_proverbs + [proverb]
    
    return Command(update={
        "proverbs": proverbs,
        "messages": [
            ToolMessage(f"Added proverb: {proverb}", tool_call_id=tool_call_id)
        ]
    })

backend_tools = [
    get_weather,
    set_name,
    generate_proverb
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
            "You are a helpful assistant with access to several tools. "
            f"The current proverbs collection contains: {state.get('proverbs', [])}. "
            f"The agent's current name is: {state.get('name', 'not set')}. "
            "\nIMPORTANT TOOL USAGE INSTRUCTIONS:\n"
            "- When the user asks to change or set the agent's name, ALWAYS call the set_name tool with the requested name.\n"
            "- When the user requests a proverb, wisdom, saying, or asks you to 'generate', 'create', 'make', or 'give' a proverb, ALWAYS call the generate_proverb tool.\n"
            "- For weather information, ALWAYS use the get_weather tool.\n"
            "- Examples of proverb requests: 'give me a proverb', 'create a proverb about love', 'generate wisdom', 'I need a saying', 'make a proverb'.\n"
            "- Always include a relevant topic when calling generate_proverb (extract from user's request or use a general theme)."
        )
    )


    # Messages are already in LangChain format, no conversion needed
    langchain_messages = list(state.get("messages", []))

    # Find the last message and check if it is a HumanMessage for learning context injection
    if langchain_messages and isinstance(langchain_messages[-1], HumanMessage):
        human_message = langchain_messages[-1]
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
            langchain_messages[-1] = HumanMessage(content=injected, additional_kwargs=getattr(human_message, "additional_kwargs", {}), response_metadata=getattr(human_message, "response_metadata", {}))
        except Exception as error:
            print(f"#################### getLearningContext error: {error}")

    # 4. Run the model to generate a response
    response = await model_with_tools.ainvoke([
        system_message,
        *langchain_messages,
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


# Define the workflow graph
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("tool_node", ToolNode(tools=backend_tools))
workflow.add_edge("tool_node", "chat_node")  # Only tool_node -> chat_node
workflow.set_entry_point("chat_node")

# Compile the workflow
graph = workflow.compile()
