import os
from langchain_groq import ChatGroq
from langchain.agents import create_react_agent
from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain import hub
from src.tools import tools

# ── LLM Setup ─────────────────────────────────────────────────────────────
def get_llm():
    return ChatGroq(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0.1
    )

# ── Agent Setup ───────────────────────────────────────────────────────────
def create_agent():
    llm = get_llm()

    # ReAct prompt from LangChain hub
    prompt = hub.pull("hwchase17/react-chat")

    # Memory to remember conversation
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    # Create ReAct agent
    agent = create_react_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )

    # Wrap with executor (handles the think→act→observe loop)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,        # shows reasoning steps
        max_iterations=5,    # prevent infinite loops
        handle_parsing_errors=True
    )

    return agent_executor


# ── Run agent ─────────────────────────────────────────────────────────────
def run_agent(agent_executor, question: str) -> str:
    try:
        response = agent_executor.invoke({"input": question})
        return response["output"]
    except Exception as e:
        return f"Agent error: {str(e)}"


if __name__ == "__main__":
    print("Creating agent...")
    agent = create_agent()

    print("\nTesting calculator tool:")
    response = run_agent(agent, "What is 15% of 847?")
    print(f"\nFinal Answer: {response}")

    print("\nTesting web search:")
    response = run_agent(agent, "What is the capital of France?")
    print(f"\nFinal Answer: {response}")