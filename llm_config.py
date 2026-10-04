import os
import streamlit as st
from crewai import LLM

MODEL_NAME = "openai/gpt-oss-120b"


def get_llm():
    api_key = st.secrets.get(
        "GROQ_API_KEY",
        os.environ.get("GROQ_API_KEY", ""),
    )

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing from Streamlit secrets."
        )

    os.environ["GROQ_API_KEY"] = api_key

    return LLM(
        model=f"groq/{MODEL_NAME}",
        api_key=api_key,
        temperature=0.2,
        max_tokens=1800,
    )
