# Nexa RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with **FastAPI**, **Streamlit**, **LangChain**, **OpenAI**, **Pinecone**, and **Redis**. The FastAPI application serves the API and launches the Streamlit interface for a local development experience.

## Features

- Chat interface with multiple conversations per user.
- Upload and index PDF, DOCX, TXT, and Markdown documents.
- Retrieve relevant document chunks through Pinecone vector search.
- Store conversation history and document metadata in Redis.
- Route questions to direct answers, document-based RAG, or OpenAI web search.
- View, search, clear, and delete chats.
- Edit and restore assistant responses.
- Configure the system prompt and conversation-context limit through the UI/API.
- FastAPI interactive API documentation.

## Architecture

```text
Browser
  |
  v
FastAPI (http://127.0.0.1:8000)
  |-- /api/* routes
  |-- serves the embedded frontend page
  |
  +---- Streamlit UI (http://127.0.0.1:8501, launched by FastAPI)
  |
  +---- OpenAI API: chat, embeddings, and web-search-capable responses
  +---- Pinecone: vector index for document chunks
  +---- Redis: chat history and document registry
```

## Requirements

- Python 3.11 or newer is recommended.
- An OpenAI API key with access to the configured models and web-search functionality.
- A Pinecone API key and an index configured for the embedding dimension.
- A Redis instance and its connection URL.
- Git, if you plan to publish the project to GitHub.

**Cost note:** this project is not fully local/offline. OpenAI, Pinecone, and hosted Redis may have usage limits or charges. Check each provider's current pricing before use. A GitHub repository does not include or provide API credits.

## Run locally on Windows PowerShell

Run these commands from the folder containing this README.

### 1. Create and activate a virtual environment

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use the environment's interpreter directly: `.\.venv\Scripts\python.exe`.

### 2. Configure environment variables

```powershell
Copy-Item .env.example .env
notepad .env
```

Replace the placeholder API keys and `REDIS_URL` with your own credentials. Do not commit `.env` or paste secrets into issues, screenshots, or public logs. For a managed Redis provider, use the provider's TLS URL when required, commonly beginning with `rediss://`.

### 3. Prepare your Pinecone index

The configured embedding model, `text-embedding-3-small`, produces 1536-dimensional vectors. Ensure the Pinecone index named by `PINECONE_INDEX` has a matching dimension. The service may create the index if it does not exist, depending on the permissions and supported configuration of your Pinecone account.

### 4. Start the application

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open:

- App: http://127.0.0.1:8000/
- API docs: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

The FastAPI lifespan starts Streamlit as a subprocess on port `8501`. You normally only need to run the `uvicorn` command. If port 8000 or 8501 is already occupied, stop the conflicting process first.

## API overview

Routes are prefixed with `/api`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/health` | Check service health / Redis connectivity |
| `POST` | `/api/documents/upload` | Upload and index a document (`multipart/form-data`) |
| `GET` | `/api/documents/{user_id}` | List a user's documents |
| `GET` | `/api/documents/{user_id}/{file_id}` | Get document details |
| `DELETE` | `/api/documents/{user_id}/{file_id}` | Remove a document |
| `POST` | `/api/chats` | Create a chat |
| `GET` | `/api/chats/{user_id}` | List chats |
| `GET` | `/api/chats/{user_id}/search?q=...` | Search chats |
| `GET` | `/api/chats/{user_id}/{chat_id}` | Get a chat |
| `POST` | `/api/chat/query` | Ask a question |
| `GET` / `PUT` / `DELETE` | `/api/config/prompt` | Read, update, or reset the system prompt |
| `GET` / `PUT` | `/api/config/context` | Read or update the context-message limit |

The full endpoint schema and request models are available at `/docs` after startup.

### Example chat request

```json
{
  "user_id": "user1",
  "question": "Summarize the main points in my uploaded documents"
}
```

Send it as JSON to `POST http://127.0.0.1:8000/api/chat/query`. To continue an existing conversation, include its returned `chat_id`.

## Project structure

```text
RAG_Chatbot_GitHub/
├── app/
│   ├── api/
│   │   └── routes.py             # FastAPI endpoints and request validation
│   ├── services/
│   │   ├── errors.py             # Service-level exceptions
│   │   ├── pinecone_service.py   # Embeddings and vector index operations
│   │   ├── rag_service.py       # Document ingestion, retrieval, and chat logic
│   │   ├── redis_service.py     # Chat history and document metadata
│   │   └── routing_service.py   # Direct / RAG / web-search routing
│   ├── ui/
│   │   └── streamlit_app.py      # Streamlit user interface
│   ├── config.py                # Environment-based settings
│   └── main.py                  # FastAPI app and local Streamlit lifecycle
├── .env.example                 # Safe configuration template
├── .gitignore
└── requirements.txt
```

## Security and deployment notes

- **Keep this project local until you add authentication and authorization.** API routes accept a caller-supplied `user_id`; that is not authentication and should not be treated as access control.
- The current `app/main.py` binds to `127.0.0.1` and configures Streamlit with CORS and XSRF protection disabled for its local embedded setup. Do not expose this configuration directly to the public internet. Revisit the frontend embedding, CORS/XSRF settings, authentication, HTTPS, rate limits, upload validation, and secret management before deployment.
- Never commit `.env`, API keys, Redis credentials, or real user documents. If a secret is accidentally committed, revoke/rotate it; deleting it from the latest commit alone is not sufficient.
- Upload size limits are configured by `MAX_UPLOAD_MB`; confirm the UI and API limits match your intended deployment.

## Troubleshooting

- **Missing settings at startup:** confirm `.env` exists and includes `OPENAI_API_KEY`, `PINECONE_API_KEY`, and `REDIS_URL`.
- **Redis connection errors:** verify the URL, credentials, TLS requirements, network access, and whether the Redis service is running.
- **Pinecone dimension errors:** make sure the index dimension matches `PINECONE_DIMENSION` and the selected embedding model.
- **OpenAI errors:** check key validity, model access, quota/billing, and the API's current availability.
- **Port already in use:** stop the process using port 8000 or 8501, or update the constants in `app/main.py` and the matching frontend configuration consistently.
- **Streamlit fails to load:** run `streamlit run app/ui/streamlit_app.py` in a separate terminal to see its error output, then check `NEXA_API_URL`.

## Current project status

This repository package contains the application source and dependency list, but no automated test suite was included in the uploaded archive. The package can be syntax-checked, but successful runtime operation still depends on valid provider credentials, service connectivity, and compatible dependency versions.
