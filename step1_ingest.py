"""
Document Ingestion Pipeline for RAG System
Processes documents from source directory into ChromaDB vector store.
"""

import os
import glob
import logging
import hashlib
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Dict, Optional, Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Import standard LangChain text parsers
from langchain_community.document_loaders import (
    TextLoader,
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config_loader import get_config


# Setup logging from configuration
config = get_config()
logging.basicConfig(
    level=getattr(logging, config.get('logging', 'level', default='INFO')),
    format=config.get('logging', 'format', default="%(asctime)s - %(levelname)s - %(message)s")
)
logger = logging.getLogger(__name__)


def get_loader_for_file(file_path: str) -> Optional[Any]:
    """Maps extensions directly to targeted open-source parsers."""
    ext = os.path.splitext(file_path)[1].lower()
    
    loader_map = {
        ".txt": TextLoader,
        ".csv": CSVLoader,
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader
    }
    
    if ext in loader_map:
        return loader_map[ext](file_path)
    return None


def process_single_file(file_path: str) -> Optional[List[Dict]]:
    """
    Worker function executed inside an isolated process.
    Parses the document text cleanly without hitting Python's GIL bottleneck.
    """
    try:
        loader = get_loader_for_file(file_path)
        if not loader:
            return None

        documents = loader.load()

        resolved_path = str(Path(file_path).resolve())
        basename = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()

        file_chunks = []
        for doc_index, doc in enumerate(documents):
            loader_metadata = dict(getattr(doc, "metadata", {}) or {})

            file_chunks.append({
                "text": doc.page_content,
                "metadata": {
                    **loader_metadata,
                    "source": basename,
                    "source_path": resolved_path,
                    "file_type": file_ext,
                    "doc_index": doc_index,
                }
            })

        return file_chunks

    except Exception as e:
        logger.error(f"? Error processing {os.path.basename(file_path)}: {str(e)}")
        return None

def chunk_documents(documents: List[Dict]) -> List[Dict]:
    """Apply text chunking strategy to documents."""
    chunk_size = config.get('chunking', 'chunk_size')
    chunk_overlap = config.get('chunking', 'chunk_overlap')

    if not chunk_size or chunk_size <= 0:
        return documents

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    all_chunks = []

    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]

        chunk_texts = splitter.split_text(text)

        for i, chunk_text in enumerate(chunk_texts):
            if chunk_text.strip():
                all_chunks.append({
                    "text": chunk_text.strip(),
                    "metadata": {
                        **metadata,
                        "chunk_index": i,
                    }
                })

    return all_chunks

def generate_stable_id(metadata: dict, chunk_text: str) -> str:
    source = metadata.get("relative_source") or metadata.get("source_path") or metadata.get("source", "unknown")
    page = metadata.get("page", metadata.get("doc_index", 0))
    chunk_index = metadata.get("chunk_index", 0)
    content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:12]
    source_hash = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"{source_hash}_p{page}_c{chunk_index}_{content_hash}"

def main():
    logger.info("🚀 Starting RAG Ingestion Pipeline...")
    
    # Load configuration paths
    paths = config.get_paths()
    source_dir = paths['source_dir']
    chroma_path = paths['chroma_path']
    local_models_dir = paths['local_models_dir']
    
    # Ensure source directory exists
    if not source_dir.exists():
        source_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Created folder '{source_dir}'. Place your files here and rerun.")
        return
    
    # Load collection settings
    collection_name = config.get('collection', 'name')
    reset_on_ingest = config.get('collection', 'reset_on_ingest')
    
    # 1. Gather all target documents matching extensions
    supported_extensions = config.get('loaders', 'enabled_extensions')
    all_files: List[str] = []
    
    for ext in supported_extensions:
        pattern = os.path.join(source_dir, "**", f"*{ext}")
        matches = glob.glob(pattern, recursive=True)
        all_files.extend(matches)
        
    total_files = len(all_files)
    if total_files == 0:
        logger.warning("⚠️ No supported files found to process.")
        return
        
    logger.info(f"📚 Detected {total_files} files inside the data directory.")

    # 2. Allocate worker processes based on configuration or auto-detect
    num_workers = config.get('advanced', 'multiprocessing_workers')
    if num_workers is None:
        num_workers = min(48, cpu_count())
    
    logger.info(f"⚙️ Spawning a Multiprocessing Pool with {num_workers} worker cores...")

    # 3. Spin up the parallel process engine
    all_parsed_chunks: List[Dict] = []
    try:
        with Pool(processes=num_workers) as pool:
            results = pool.map(process_single_file, all_files)
            
            for res in results:
                if res:
                    all_parsed_chunks.extend(res)
    except Exception as e:
        logger.error(f"❌ Error during multiprocessing: {str(e)}")
        return

    # 4. Apply chunking strategy to parsed documents
    strategy = config.get('chunking', 'strategy')
    if strategy != "recursive":
        raise ValueError(f"Unsupported chunking strategy for v0.1: {strategy}")

    if strategy:
        logger.info("✂️ Applying text chunking strategy...")
        all_parsed_chunks = chunk_documents(all_parsed_chunks)
        
    logger.info(f"📄 Text extraction complete. Generated {len(all_parsed_chunks)} unified text chunks.")

    # 5. Batch Vectorization & Ingestion into ChromaDB
    logger.info("🧠 Initializing Local Embedding Engine and Connecting to ChromaDB...")
    
    chroma_client = chromadb.PersistentClient(path=str(chroma_path))
    
    embedding_model_path = config.get('embedding', 'model_name')
    embedding_device = config.get('embedding', 'device')

    if not embedding_model_path:
        raise ValueError("embedding.model_name is required in config.")

    if os.path.isabs(embedding_model_path):
        raise ValueError(
            "embedding.model_name must be a local folder name under paths.local_models_dir, "
            "not an absolute path."
        )

    full_model_path = str(local_models_dir / Path(embedding_model_path).name)

    if not Path(full_model_path).exists():
        raise FileNotFoundError(
            f"Embedding model folder not found: {full_model_path}\n"
            f"Expected embedding.model_name to name a folder under paths.local_models_dir."
        )

    logger.info(f"?? Using embedding model: {full_model_path}")
    logger.info(f"?? Embedding device: {embedding_device}")

    bge_embedding = SentenceTransformerEmbeddingFunction(
        model_name=full_model_path, 
        device=embedding_device
    )
    
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=bge_embedding
    )
    
    # Optionally reset collection if configured
    if reset_on_ingest:
        logger.info("🔄 Resetting existing collection as per configuration...")
        chroma_client.delete_collection(collection_name)
        collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=bge_embedding
        )

    # Deconstruct payloads into flat ingestion formats
    texts = [chunk["text"] for chunk in all_parsed_chunks]
    metadatas = [chunk["metadata"] for chunk in all_parsed_chunks]
    ids = [
        generate_stable_id(chunk["metadata"], chunk["text"])
        for chunk in all_parsed_chunks
    ]
    logger.info(f"Generated {len(ids)} unique document IDs.")
    logger.info("Committing records into ChromaDB database layers...")
    chroma_batch_size = config.get("ingestion", "chroma_batch_size") or 1000
    for i in range(0, len(texts), chroma_batch_size):
        end_idx = min(i + chroma_batch_size, len(texts))
        collection.upsert(
            documents=texts[i:end_idx],
            metadatas=metadatas[i:end_idx],
            ids=ids[i:end_idx],
        )
        logger.info(f"  Processed progress status: {end_idx} / {len(texts)} chunks stored.")
    logger.info("✅ System successfully ingested your multi-format directory at peak computing capability!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("⚠️ Ingestion interrupted by user.")
    except Exception as e:
        logger.error(f"❌ Fatal error during ingestion: {str(e)}")
