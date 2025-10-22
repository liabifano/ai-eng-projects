from openai import OpenAI
import requests
import inspect
from langchain_core.tools import tool
from langchain_community.chat_models import ChatOllama
from langchain.agents import initialize_agent, Tool, AgentType
from duckduckgo_search import DDGS

client = OpenAI(api_key = "ollama", base_url = "http://localhost:11434/v1")

OPENWEATHERMAP_API_KEY = ""

def get_current_weather(city: str, unit: str = "celsius") -> str:
    """Retuns weather forecast from openweathermap.org

    @city: str City
    @unit: str Temperature Unit (default celsius)
    """

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    complete_url = f"{base_url}&appid={OPENWEATHERMAP_API_KEY}&?q={city}"
    response = requests.get(complete_url).json()
    # return response
    return "It will be sunny, 23C, won't rain"

@tool
def get_weather(city: str) -> str:
    """
    Retuns weather forecast from openweathermap.org
    """
    return get_current_weather(city)


def select_and_call_webtools(USER_QUERY: str) -> dict:
    if USER_QUERY.find('is the weather'):
        city = " ".join([x for x in USER_QUERY.split(" ") if x.istitle()][1:])
        return {"get_current_wheater": get_current_weather(city)}
    
    else: return {}

@tool
def web_search(query: str) -> str:
    """
    Search in the web using duckduckgo
    """
    results = DDGS().text(query, max_results=5)
    return "\n\n".join([f"{r['title']} - ref:{r['href']} - {r["body"]}" for r in results])


def to_schema(fn):
    return {"name": fn.name,
            "func": fn,
            "description": fn.description,
            "args": list(inspect.signature(fn).parameters.keys()),
            "types": [p.annotation for p in list(inspect.signature(fn).parameters.values())]
            }


if __name__ == '__main__':
    city = "San Francisco"
    USER_QUERY = f"""What is the weather in SF today?"""

    TOOLS_SCHEMA = {"get_weather": to_schema(get_weather), "web_search": to_schema(web_search)}
    tools = [Tool(**t) for t in TOOLS_SCHEMA.values()]

    SYSTEM_PROMPT = f"""
    User asked: {USER_QUERY}
    Here is the tools you can choose from and execute to answer user's question. 
        {TOOLS_SCHEMA}

    Do the steps in the following order: 
        1) Always select web search in TOOLS_SCHEMA with the whole user query
        2) In TOOLS_SCHEMA, find additional tools that matches or answers user's query. 
        3) Map the "args" of the matched tools in step 1. Use default if user didn't specify.
        4) If any argument is missing and there is no default, ask to the user include it in the query
        5) Execute the tools 
        6) Summarize tools results, reason the user's query and the returned values from the tools, provide an empatetic and clear answer to the final user
    """

    model = ChatOllama(model="gemma3:1b", temperature=0)

    agent = initialize_agent(tools=tools, llm=model)

    output = agent.invoke(SYSTEM_PROMPT, verbose=True)['output']

    print(output)
