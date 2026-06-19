"""
RAG Application Server with Streaming Support
Handles queries, retrieval, and LLM communication via LM Studio.
"""

import os
import re
import asyncio
import logging
from pathlib import Path  
from typing import List, Dict, Any, Optional

import chromadb
import tiktoken
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from openai import AsyncOpenAI

from config_loader import get_config


# Setup logging from configuration
config = get_config()
logging.basicConfig(
    level=getattr(logging, config.get('logging', 'level', default='INFO')),
    format=config.get('logging', 'format', default="%(asctime)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Inveate API",
    description=f"RAG Pipeline Version {config.get('collection', 'version')}"
)

# Add CORS if enabled in config
if config.get('server', 'cors_enabled'):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class QueryRequest(BaseModel):
    query: str
    history: List[Dict[str, str]] = Field(default_factory=list)

# --- ENGINE PROFILE CONFIGURATION (Loaded from config.yaml) ---
CHROMA_PATH = config.get_paths()['chroma_path']
LOCAL_MODEL_PATH = os.path.join(
    config.get_paths()['local_models_dir'],
    Path(config.get('embedding', 'model_name')).name  
)

# Context management settings
MAX_SYSTEM_CONTEXT = config.get('context', 'max_system_context')
THINKING_TOKEN_BUFFER = config.get('context', 'reserved_generation_tokens')
CONTEXT_SEPARATOR = config.get('context', 'separator')

# LM Studio connection settings
LM_BASE_URL = config.get('lmstudio', 'base_url')
LM_API_KEY = config.get('lmstudio', 'api_key')
LM_MODEL = config.get('lmstudio', 'model')
LM_TEMPERATURE = config.get('lmstudio', 'temperature')
LM_MAX_GEN_TOKENS = config.get('lmstudio', 'max_gen_tokens')


# Initialize background connections at server boot
logger.info("🗄️ Connecting to optimized ChromaDB storage engine...")
chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))

embedding_model_path = LOCAL_MODEL_PATH if not os.path.isabs(LOCAL_MODEL_PATH) else LOCAL_MODEL_PATH
bge_embedding_function = SentenceTransformerEmbeddingFunction(
    model_name=embedding_model_path,
    device=config.get('embedding', 'device')
)

collection_name = config.get('collection', 'name')

try:
    collection = chroma_client.get_collection(
        name=collection_name, 
        embedding_function=bge_embedding_function
    )
except Exception as e:
    logger.error(f"❌ Failed to connect to ChromaDB collection '{collection_name}': {str(e)}")
    raise

# Connect to LM Studio Server
lm_studio_client = AsyncOpenAI(
    base_url=LM_BASE_URL, 
    api_key=LM_API_KEY
)


def calculate_token_count(text_string: str) -> int:
    """Calculates token weight locally using configured tokenizer."""
    encoder_name = config.get('advanced', 'token_encoder') or "cl100k_base"
    
    try:
        encoding = tiktoken.get_encoding(encoder_name)
        return len(encoding.encode(text_string))
    except Exception as e:
        logger.warning(f"⚠️ Token counting error ({e}), using approximate character-based estimate")
        # Fallback estimation (rough approximation)
        return int(len(text_string) / 4)

def check_api_key(request: Request):
    expected = config.get("server", "api_key")
    if not expected:
        return
    supplied = request.headers.get("x-api-key")
    if supplied != expected:
        raise HTTPException(status_code=401, detail="Invalid API key")

def enforce_context_guardrail(
    system_prompt: str, 
    historical_text: str, 
    context_chunks: List[str]
) -> tuple[str, int]:
    """
    Dynamically trims context chunks to protect the GPU thinking budget.
    Returns (safe_context_block, token_count).
    This ensures consistent behavior between client and server trimming logic.
    """
    if not config.get('context', 'enable_guardrail'):
        return CONTEXT_SEPARATOR.join(context_chunks), calculate_token_count(
            f"{system_prompt}\n{historical_text}\n{CONTEXT_SEPARATOR.join(context_chunks)}"
        )
        
    max_allowed_prompt_budget = MAX_SYSTEM_CONTEXT - THINKING_TOKEN_BUFFER
    
    working_chunks = context_chunks.copy()
    
    while len(working_chunks) > 0:
        test_context_block = CONTEXT_SEPARATOR.join(working_chunks)
        full_test_string = f"{system_prompt}\n{historical_text}\n{test_context_block}"
        
        current_token_weight = calculate_token_count(full_test_string)
        
        if current_token_weight <= max_allowed_prompt_budget:
            logger.info(f"🛡️ Guardrail Verified: Input payload is {current_token_weight} tokens.")
            return test_context_block, current_token_weight
        
        working_chunks.pop()

    # Fallback to system message only
    fallback_string = "No relevant data context found for this query."
    final_fallback_tokens = calculate_token_count(f"{system_prompt}\n{historical_text}\n{fallback_string}")
    
    logger.warning(f"🛡️ Guardrail Alert: History exceeded budget. Using fallback ({final_fallback_tokens} tokens).")
    
    return fallback_string, final_fallback_tokens

@app.post("/query", response_model=None)
async def handle_rag_query(query_request: QueryRequest, http_request: Request):
    """
    Accepts a query, pulls from ChromaDB, audits context, 
    and streams tokens directly to the client.
    """
    check_api_key(http_request)
    user_query = query_request.query.strip()
    
    if not user_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Fetch data from optimized database index
    top_k = config.get('retrieval', 'top_k') or 6
    
    loop = asyncio.get_event_loop()
    
    try:
        results = await loop.run_in_executor(
            None, 
            lambda: collection.query(
                query_texts=[user_query], 
                n_results=top_k,
                include=["documents", "metadatas"]
            )
        )
        
        # Flatten ChromaDB's multi-nested list structure into a flat list of plain strings
        raw_context_chunks: List[str] = []
        
        if results and 'documents' in results and results['documents']:
            for sublist in results['documents']:
                for document_text in sublist:
                    if document_text:
                        raw_context_chunks.append(str(document_text).strip())

    except Exception as e:
        logger.error(f"❌ Retrieval error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Database retrieval failed: {str(e)}")
    
    # Protect against an empty database return
    if not raw_context_chunks:
        raw_context_chunks = ["No relevant records found in ChromaDB matching this query identifier."]

    # 2. Extract and format conversational history text
    chat_history = query_request.history or []
    
    # Apply history trimming before token counting for guardrail
    max_turns = config.get('chat', 'max_turns') or 12
    
    # Trim to match client behavior (max_turns * 2 = raw message count)
    if len(chat_history) > max_turns * 2:
        chat_history = chat_history[-(max_turns * 2):]
    
    history_text = "".join([f"\n{turn['role']}: {turn['content']}" for turn in chat_history])

    # 3. Apply the Token-Counter Guardrail (now with trimmed history)
    system_instruction = (
        "You are an elite reasoning assistant. Answer the user's question using the provided CSV and document data context. "
        "Analyze the data systematically inside your thoughts before outputting your final response."
    )
    
    safe_context_block, prompt_tokens = enforce_context_guardrail(
        system_prompt=system_instruction,
        historical_text=history_text,
        context_chunks=raw_context_chunks
    )

    # 4. Construct the complete payload (using already-trimmed chat_history)
    injected_prompt = f"""Use the following data context to answer the question at the end.
    
[CONTEXT]
{safe_context_block}

[QUESTION]
{user_query}"""

    messages_payload: List[Dict[str, str]] = [{"role": "system", "content": system_instruction}]
    
    # Add chat history (already trimmed above)
    for turn in chat_history:
        messages_payload.append(turn)
        
    messages_payload.append({"role": "user", "content": injected_prompt})

    # 5. Define the asynchronous streaming generator
    async def token_stream_generator():
        logger.info(f"\n🧠 Sending payload to LM Studio ({prompt_tokens} prompt tokens loaded)...")
        
        try:
            response_stream = await lm_studio_client.chat.completions.create(
                model=LM_MODEL,
                messages=messages_payload,
                temperature=LM_TEMPERATURE,
                max_tokens=LM_MAX_GEN_TOKENS,
                stream=True
            )

            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    token = chunk.choices[0].delta.content
                    yield token
                    
        except Exception as e:
            logger.error(f"❌ LM Studio streaming error: {str(e)}")
            yield f"[ERROR] Model inference failed: {str(e)}"

    return StreamingResponse(
        token_stream_generator(), 
        media_type=config.get('server', 'stream_format') or "text/plain",
        headers={"Cache-Control": "no-cache"}
    )


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {
        "status": "healthy",
        "collection": config.get('collection', 'name'),
        "model": LM_MODEL,
        "chroma_path": str(CHROMA_PATH),
        "app": config.get("app", "name") or "Inveate",
        "version": config.get("app", "version") or "0.1.0"
    }


@app.on_event("startup")
async def startup_event():
    """Log server configuration on boot."""
    logger.info("=" * 60)
    logger.info(f"🚀 Inveate RAG Engine Starting...")
    logger.info(f"   ChromaDB: {CHROMA_PATH}")
    logger.info(f"   Collection: {collection_name}")
    logger.info(f"   LM Studio: {LM_BASE_URL}")
    logger.info(f"   Model: {LM_MODEL}")
    logger.info("=" * 60)


if __name__ == "__main__":
    server_config = config.get('server')
    print("🚀 Booting Async Orchestration Server")
    uvicorn.run(
        app, 
        host=server_config['host'], 
        port=server_config['port'],
        log_level=config.get('logging', 'level', default='info').lower()
    )
