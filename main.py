import os
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

PUBLIC_DIR = BASE_DIR / "public"
STATIC_DIR = PUBLIC_DIR / "static"
KNOWLEDGE_FILE = BASE_DIR / "sample_docs" / "knowledge.txt"


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")


# =========================================================
# CHECK API KEY
# =========================================================

if not API_KEY:
    raise ValueError(
        "GROQ_API_KEY is missing. "
        "Please add it to your .env file."
    )


# =========================================================
# GROQ CLIENT
# =========================================================

client = Groq(api_key=API_KEY)


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Groq AI Chat",
    description="Groq AI Chat with Basic Chat and RAG",
    version="1.0.0"
)


# =========================================================
# STATIC FILES
# =========================================================

if STATIC_DIR.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(STATIC_DIR)),
        name="static"
    )


# =========================================================
# REQUEST MODEL
# =========================================================

class ChatRequest(BaseModel):
    message: str
    mode: str = "basic"


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/")
async def home():

    index_file = PUBLIC_DIR / "index.html"

    if not index_file.exists():
        return {
            "error": "public/index.html not found"
        }

    return FileResponse(str(index_file))


# =========================================================
# BASIC CHAT
# =========================================================

def basic_chat(message: str):

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": message
            }
        ]
    )

    return response.choices[0].message.content


# =========================================================
# RAG CHAT
# =========================================================

def rag_chat(question: str):

    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_FILE}"
        )

    with open(
        KNOWLEDGE_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        document = file.read()

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful RAG assistant. "
                    "Answer the user's question using ONLY "
                    "the provided document. "
                    "If the answer is not available in the "
                    "document, clearly say that the information "
                    "is not available in the document."
                )
            },
            {
                "role": "user",
                "content": (
                    f"DOCUMENT:\n\n{document}\n\n"
                    f"QUESTION:\n\n{question}"
                )
            }
        ]
    )

    return response.choices[0].message.content


# =========================================================
# CHAT API
# =========================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    message = request.message.strip()

    if not message:
        return {
            "success": False,
            "reply": "Please enter a message."
        }

    mode = request.mode.lower().strip()

    try:

        if mode == "basic":

            reply = basic_chat(message)

            return {
                "success": True,
                "mode": "basic",
                "reply": reply
            }

        elif mode == "rag":

            reply = rag_chat(message)

            return {
                "success": True,
                "mode": "rag",
                "reply": reply
            }

        else:

            return {
                "success": False,
                "reply": "Invalid mode. Use Basic Chat or RAG Chat."
            }

    except FileNotFoundError as error:

        return {
            "success": False,
            "reply": str(error)
        }

    except Exception as error:

        print("ERROR:", error)

        return {
            "success": False,
            "reply": (
                "Something went wrong while contacting "
                "the Groq API. Please check your API key "
                "and model name."
            )
        }


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
async def health():

    return {
        "status": "ok",
        "model": MODEL
    }