from openai import OpenAI
import requests
import inspect

client = OpenAI(api_key = "ollama", base_url = "http://localhost:11434/v1")

OPENWEATHERMAP_API_KEY = ""


def get_current_weather(city: str, unit: str = "celsius") -> str:
    """ Retuns weather forecast from openweathermap.org

    @city: str City
    @unit: str Temperature Unit (default celsius)
    """

    base_url = "http://api.openweathermap.org/data/2.5/weather"
    complete_url = f"{base_url}&appid={OPENWEATHERMAP_API_KEY}&?q={city}"
    response = requests.get(complete_url).json()
    # return response
    return "It's 23C and sunny"


def select_and_call_webtools(USER_QUERY: str) -> dict:
    if USER_QUERY.find('is the weather'):
        city = " ".join([x for x in USER_QUERY.split(" ") if x.istitle()][1:])
        return {"get_current_wheater": get_current_weather(city)}
    
    else: return {}
    

TOOLS = {"weather": get_current_weather}

def to_schema(fn):
    return {"name": fn.__name__,
            "description": fn.__doc__,
            "args": list(inspect.signature(fn).parameters.keys()),
            "types": [p.annotation for p in list(inspect.signature(fn).parameters.values())]
            }


if __name__ == '__main__':
    city = "San Francisco"
    USER_QUERY = f"""What is the weather in {city} today?"""

    TOOLS_SCHEMA = {"get_current_weather": to_schema(get_current_weather)}

    SYSTEM_PROMPT = f"""
    User asked: {USER_QUERY}
    Here is the tools you can choose from and execute to answer user's question: 
        {TOOLS_SCHEMA}

    Do the steps in the following order: 
        1) In TOOLS_SCHEMA, find a tools that matches or answers user's query
        2) Map the "args" of the matched tools in step 1
        3) If any argument is missing and there is no default, ask to the user include it in the query
        4) Provide which tools and arguments its required to answer user's query. 
            The output must be a list of dictionaries where key are tools' names and values are dictionaries
            where keys are args names and values are args values. 
    """

    stream = client.chat.completions.create(
        model="mistral:7b",
        messages=[{
            "role": "system",
            "content": "You are a helpful assistant that uses retrieved webdata to answer questions"
            }, {
                "role": "user",
                "content": SYSTEM_PROMPT
            }, 
            ],
        stream=True,
    )

    for chunk in stream:
        print(chunk.choices[0].delta.content or "", end="")
    