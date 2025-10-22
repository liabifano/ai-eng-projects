from openai import OpenAI
import requests

client = OpenAI(api_key = "ollama", base_url = "http://localhost:11434/v1")

OPENWEATHERMAP_API_KEY = ""


def get_current_wheater(city: str, unit: str = "celsius") -> str:
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    complete_url = f"{base_url}&appid={OPENWEATHERMAP_API_KEY}&?q={city}"
    response = requests.get(complete_url).json()
    # return response
    return "It's 23C and sunny"


def select_and_call_webtools(USER_QUERY: str) -> dict:
    if USER_QUERY.find('is the weather'):
        city = " ".join([x for x in USER_QUERY.split(" ") if x.istitle()][1:])
        return {"get_current_wheater": get_current_wheater(city)}
    
    else: return {}
    

if __name__ == '__main__':
    city = "San Francisco"
    USER_QUERY = f"""What is the weather in {city} today?"""

    web_response = select_and_call_webtools(USER_QUERY)

    SYSTEM_PROMPT = f"""
    User asked: {USER_QUERY}
    Here is the live info from the web: 
        {web_response}
        
    Provide a helpful, clear answer.

    Rules:
        1) Check if the city exists and there is no misspelling. If the checks do not pass, ask the user to check its inputs and re-try
        2) Be concise and accurate. Prefer quoting key phrases from the context.
        3) When possible and trustable, cite sources as using the metadata

    If the answer is not in CONTEXT or updated, respond with “I'm not sure from the docs.
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
    