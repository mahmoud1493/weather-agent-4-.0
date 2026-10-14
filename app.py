import os
import uuid

import requests
import streamlit as st
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langgraph.checkpoint.postgres import PostgresSaver


load_dotenv()


def load_streamlit_secrets_into_environment():
    """Copy Streamlit Cloud secrets into environment variables once at startup."""
    required_keys = [
        "GROQ_API_KEY",
        "OPENWEATHER_API_KEY",
        "SUPABASE_DB_URI",
    ]

    for key in required_keys:
        try:
            value = st.secrets.get(key)

            if value:
                os.environ[key] = str(value)
        except Exception:
            # Local development uses values loaded from .env.
            pass


load_streamlit_secrets_into_environment()

st.set_page_config(
    page_title="Weather Assistant",
    page_icon="☀️",
    layout="centered",
)


SYSTEM_PROMPT = """
You are a helpful weather assistant.

YOUR WORKFLOW:

1. If the user asks about weather WITHOUT specifying a location, you MUST:
   - First call get_location() to find the user's location.
   - Then call get_weather(city) using the location returned by get_location().
2. If the user provides a city, call get_weather(city) directly.
3. Clearly explain the weather conditions and temperature.
4. Include both Celsius and Fahrenheit when temperature data is available.
"""


def get_secret(name: str) -> str | None:
    """Read a value loaded from .env or Streamlit Cloud secrets."""
    value = os.getenv(name)
    return value.strip() if value else None


def require_secret(name: str) -> str:
    value = get_secret(name)

    if not value:
        raise RuntimeError(
            f"Missing {name}. Add it to .env locally or to Streamlit "
            "Community Cloud app Secrets."
        )
    return value


def get_weather(city: str):
    """Get current weather for a city from OpenWeatherMap."""
    api_key = require_secret("OPENWEATHER_API_KEY")
    base_url = "https://api.openweathermap.org/data/2.5/weather"

    response = requests.get(
        base_url,
        params={
            "q": city,
            "appid": api_key,
            "units": "metric",
        },
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    temperature_celsius = float(data["main"]["temp"])
    temperature_fahrenheit = temperature_celsius * 9 / 5 + 32

    return {
        "city": data.get("name", city),
        "country": data.get("sys", {}).get("country"),
        "condition": data.get("weather", [{}])[0].get("description"),
        "temperature_celsius": round(temperature_celsius, 1),
        "temperature_fahrenheit": round(temperature_fahrenheit, 1),
        "humidity_percent": data.get("main", {}).get("humidity"),
        "wind_speed_meters_per_second": data.get("wind", {}).get("speed"),
    }


def get_location():
    """Estimate the server's public-IP location when no city is provided."""
    response = requests.get(
        "https://ipapi.co/json/",
        headers={"User-Agent": "weather-assistant/1.0"},
        timeout=15,
    )
    response.raise_for_status()

    data = response.json()
    city = data.get("city")
    country = data.get("country_name")

    if not city:
        raise RuntimeError("Could not determine a location automatically.")

    return f"{city}, {country}" if country else city


def build_agent():
    """Build one agent/checkpointer pair for the current request."""
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0.7,
        api_key=require_secret("GROQ_API_KEY"),
    )

    db_uri = require_secret("SUPABASE_DB_URI")
    checkpointer_context = PostgresSaver.from_conn_string(db_uri)
    checkpointer = checkpointer_context.__enter__()
    checkpointer.setup()

    agent = create_agent(
        model=llm,
        tools=[get_weather, get_location],
        system_prompt=SYSTEM_PROMPT,
        checkpointer=checkpointer,
    )

    return agent, checkpointer_context


def run_agent(user_query: str, thread_id: str) -> str:
    """Run the agent and close the database connection after the request."""
    agent, checkpointer_context = build_agent()

    try:
        response = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": user_query,
                    }
                ]
            },
            {
                "configurable": {
                    "thread_id": thread_id,
                }
            },
        )
    finally:
        checkpointer_context.__exit__(None, None, None)

    final_message = response["messages"][-1]
    content = final_message.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                text_parts.append(block["text"])
            elif block:
                text_parts.append(str(block))
        return "\n".join(text_parts)

    return str(content)


st.title("Weather Assistant")
st.caption("Ask about the weather in any city.")

if "thread_id" not in st.session_state:
    # A unique ID prevents different visitors from sharing one conversation.
    st.session_state.thread_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me about the weather in a city.",
        }
    ]

with st.sidebar:
    st.header("Conversation")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is the weather in Rome?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Checking the weather..."):
            try:
                answer = run_agent(prompt, st.session_state.thread_id)
            except requests.HTTPError as exc:
                answer = (
                    "I could not retrieve the weather service response. "
                    f"Please check the city name and API key. Details: {exc}"
                )
            except Exception as exc:
                answer = f"Something went wrong: {exc}"

        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )
