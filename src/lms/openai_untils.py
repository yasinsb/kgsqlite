import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file if it exists
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError(
        "OPENAI_API_KEY not found in environment variables. "
        "Please set it in .env file or as an environment variable."
    )

from openai import OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"



def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model=EMBEDDING_MODEL
    )
    return response.data[0].embedding

def embed_text_batch(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model=EMBEDDING_MODEL
    )
    return [data.embedding for data in response.data]