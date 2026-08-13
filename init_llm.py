import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    """Initializes and returns the Gemini model."""
    return ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", temperature=0.0)