"""
CLI Client Interface for RAG Pipeline
Handles user interaction with streaming responses from the server.
"""

import os
import re
import json
import sys
import time
import threading
from typing import List, Dict, Optional
from pathlib import Path

import requests

from config_loader import get_config


# Setup from configuration
config = get_config()
SERVER_URL = config.get('client', 'server_url') or "http://127.0.0.1:8000/query"
STREAM_ENABLED = config.get('client', 'stream')
STRIP_THINKING = config.get('chat', 'strip_thinking_blocks')

# Thinking block tags - support multiple patterns 
THINKING_TAGS = config.get('chat', 'thinking_tags') or [
    {"start": "<think>", "end": "</think>"},
    {"start": "<thought>", "end": "</thought>"}
]

# Session persistence settings
PERSIST_SESSIONS = config.get('chat', 'persist_sessions') or False
SESSION_PATH = Path(config.get('chat', 'session_path', default="./sessions/"))

class ActivityIndicator:
    """Simple terminal activity indicator for long blocking waits."""

    def __init__(self, message: str, interval: float = 1.0):
        self.message = message
        self.interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        print(self.message, end="", flush=True)

        while not self._stop_event.wait(self.interval):
            print(".", end="", flush=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        print("", flush=True)  # finish the status line
        
        
def load_session() -> List[Dict[str, str]]:
    """Load conversation history from disk if persistence is enabled."""
    if not PERSIST_SESSIONS:
        return []
        
    SESSION_PATH.mkdir(parents=True, exist_ok=True)
    session_file = SESSION_PATH / "conversation_history.json"
    
    if session_file.exists():
        try:
            with open(session_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
                print(f"📂 Loaded {len(history)} messages from previous session.") 
                return history
        except Exception as e:
            print(f"[WARNING] Could not load session: {e}")
            
    return []


def save_session(history: List[Dict[str, str]]) -> None:
    """Save conversation history to disk if persistence is enabled."""
    if not PERSIST_SESSIONS:
        return
        
    SESSION_PATH.mkdir(parents=True, exist_ok=True)
    session_file = SESSION_PATH / "conversation_history.json"
    
    try:
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[WARNING] Could not save session: {e}")


def clean_response(raw_text: str) -> str:
    """Remove thinking/reasoning blocks from assistant output."""
    if not STRIP_THINKING:
        return raw_text
    
    cleaned = raw_text
    
    #Iterate through all configured thinking tag patterns
    for tag_config in THINKING_TAGS:
        start_tag = tag_config.get('start', '<thought>')
        end_tag = tag_config.get('end', '</thought>')
        
        if not start_tag or not end_tag:
            continue
            
        pattern = re.escape(start_tag) + r".*?" + re.escape(end_tag)
        
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL)
    
    # Also strip any remaining standalone opening tags (defensive cleanup)
    for tag_config in THINKING_TAGS:
        start_tag = tag_config.get('start', '<thought>')
        if start_tag:
            pattern_no_end = re.escape(start_tag) + r".*?(?=\n|$)"
            cleaned = re.sub(pattern_no_end, '', cleaned, flags=re.DOTALL)
    
    return cleaned.strip()


def main():
    APP_NAME = config.get("app", "name") or "Inveate"
    APP_VERSION = config.get("app", "version") or "0.1.0"
    print("\n" + "=" * 60)
    print(f"{APP_NAME} v{APP_VERSION} RAG Interactive Chat Client")
    print("=" * 60)
    print(f"Server: {SERVER_URL}")
    print(f"Streaming: {'Enabled' if STREAM_ENABLED else 'Disabled'}")
    print(f"Thinking Blocks: {'Stripped' if STRIP_THINKING else 'Preserved'}")
    print("Type your prompt below. Type 'exit' or '/clear' to manage session.\n")

    # Load previous conversation history if persistence enabled
    chat_history = load_session()
    
    # Apply max turns limit (treats as raw message count for consistency with server)
    max_turns = config.get('chat', 'max_turns') or 12
    
    #Trim to match server behavior (messages, not pairs)
    if len(chat_history) > max_turns * 2:
        chat_history = chat_history[-(max_turns * 2):]

    while True:
        query = input("\n❓ Prompt: ").strip()
        answer_header_pending = True
        if not query:
            continue
            
        # Session management commands
        if query.lower() == 'exit':
            print("\n👋 Goodbye!")
            break
            
        if query.lower() == '/clear':
            chat_history = []
            save_session(chat_history)
            print("💡 Conversation history cleared.")
            continue

        payload = {
            "query": query,
            "history": chat_history
        }

        full_response_text = ""

        headers = {}
        api_key = config.get("server", "api_key")
        if api_key:
            headers["x-api-key"] = api_key

        indicator = ActivityIndicator("⏳ Inveate toolchain is processing")
        indicator.start()

        answer_header_pending = True

        try:
            response = requests.post(
                SERVER_URL,
                json=payload,
                headers=headers,
                stream=STREAM_ENABLED,
                timeout=config.get("client", "connection_timeout", default=30)
            )

            if response.status_code != 200:
                indicator.stop()
                error_msg = response.text or "Unknown error"
                print(f"❌ Server Error ({response.status_code}): {error_msg}")
                continue

            if STREAM_ENABLED and config.get("server", "stream_format") == "text/plain":
                for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                    if chunk:
                        if answer_header_pending:
                            indicator.stop()
                            print("💡 ANSWER:")
                            answer_header_pending = False

                        print(chunk, end="", flush=True)
                        full_response_text += chunk

                if answer_header_pending:
                    indicator.stop()
                    print("⚠️ Server returned no streamed content.")

            else:
                full_response_text = response.text
                indicator.stop()

                if full_response_text:
                    print("💡 ANSWER:")
                    print(full_response_text)
                else:
                    print("⚠️ Server returned an empty response.")

                
            print()  # Print newline at completion

        except requests.exceptions.Timeout:
            print("\n❌ Request timed out. Please try again.")
            continue
        except requests.exceptions.ConnectionError:
            print(f"\n❌ Connection Error. Is server running at {SERVER_URL}?")
            continue
        except Exception as e:
            print(f"\n❌ Client Error: {str(e)}")
            continue

        # Clean and save the response to history (only reached on success now)
        clean_history_text = clean_response(full_response_text)
        
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": clean_history_text})
        
        # Apply max turns limit before saving 
        if len(chat_history) > max_turns * 2:
            chat_history = chat_history[-(max_turns * 2):]
            
        save_session(chat_history)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Client interrupted by user.")
