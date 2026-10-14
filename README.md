# 🌦️ Weather Agent

**Live app:** [Open Weather Agent](https://weather-agent-2-mi7773.streamlit.app/)

AI weather assistant built with **LangChain, LangGraph, Groq, OpenWeatherMap, Supabase, and Streamlit**.

## Run locally

```bash
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Add these values to `.env`:

```
GROQ_API_KEY=your_groq_api_key
OPENWEATHER_API_KEY=your_openweathermap_api_key
SUPABASE_DB_URI="your_supabase_postgresql_uri"
```

Never upload `.env` or real API keys to GitHub.
