import os

from dotenv import load_dotenv
from groq import Groq

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from pydantic import BaseModel


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

KNOWLEDGE_FILE = "sample_docs/knowledge.txt"


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

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(
    directory="templates"
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

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )


# =========================================================
# BASIC CHAT FUNCTION
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
# RAG CHAT FUNCTION
# =========================================================

def rag_chat(question: str):

    if not os.path.exists(KNOWLEDGE_FILE):
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
                    f"DOCUMENT:\n\n"
                    f"{document}\n\n"
                    f"QUESTION:\n\n"
                    f"{question}"
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

        # -----------------------------------------
        # BASIC CHAT
        # -----------------------------------------

        if mode == "basic":

            reply = basic_chat(message)

            return {
                "success": True,
                "mode": "basic",
                "reply": reply
            }


        # -----------------------------------------
        # RAG CHAT
        # -----------------------------------------

        elif mode == "rag":

            reply = rag_chat(message)

            return {
                "success": True,
                "mode": "rag",
                "reply": reply
            }


        # -----------------------------------------
        # INVALID MODE
        # -----------------------------------------

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