"""
RAG system configuration - centralizes all settings
"""
from pathlib import Path
from typing import Dict, Any
import logging
import hashlib

from ..utils.llm_runtime import derive_bounded_stateful_session_id, invoke_with_retry

logger = logging.getLogger(__name__)

# ==================== Path configuration ====================
def get_project_root() -> Path:
    """Return the project root directory"""
    current_file = Path(__file__).resolve()
    return current_file.parent.parent.parent

def get_rag_dir() -> Path:
    """Return the RAG directory"""
    return Path(__file__).parent.resolve()

PROJECT_ROOT = get_project_root()
RAG_DIR = get_rag_dir()

# ==================== Directory layout ====================
DIRS = {
    "literature_db": RAG_DIR / "literature_db",
    "literature": RAG_DIR / "literature_db" / "literature",
    "knowledge_base": RAG_DIR / "knowledge_base",
    "embedding_base": RAG_DIR / "embedding_base",
    "output": PROJECT_ROOT / "cytobridge_output",
}

# ==================== File paths ====================
FILES = {
    "apa_citations": DIRS["literature_db"] / "apa.txt",
    "structured_knowledge_base": DIRS["knowledge_base"] / "structured_knowledge_base.json",
    "kb_version": DIRS["knowledge_base"] / ".kb_version",
    "pdf_mod_times": DIRS["knowledge_base"] / ".pdf_mod_times.json",
    "paragraph_index": DIRS["embedding_base"] / "paragraph_index.json",
    "vector_index": DIRS["embedding_base"] / "vector_index.npy",
}

# ==================== Model configuration ====================
MODEL_CONFIG = {
    # Sentence TransformerModel configuration
    "sentence_transformer": {
        "model_name": "sentence-transformers/all-mpnet-base-v2",  # Model name; either a HuggingFace model name or a local path
        "local_path": RAG_DIR / "all-mpnet-base-v2",  # Local model path
        "device": "cpu",  # Runtime device: cpu or cuda
        "batch_size": 32,  # Encoding batch size
    },
    
    # spaCyModel configuration
    "spacy": {
        "model_name": str(RAG_DIR / "en_core_web_sm-3.8.0"),
        "disable_components": [],
        "fallback_to_regex": True,
    },
    
    # Additional model configuration can be added here
    "alternative_models": {
        "mini": "all-MiniLM-L6-v2",  # Smaller and faster model
        "large": "all-mpnet-base-v2",  # Default model
    }
}

# ==================== Retrieval configuration ====================
SEARCH_CONFIG = {
    "min_relevance_score": 0.3,
    "max_search_results": 20,
    "rrf_k_constant": 60,
    
    # Retrieval strategy
    "hyde_enabled": True,
    "rerank_enabled": True,
    
    # Cache configuration
    "cache_enabled": True,
    "cache_size": 100,
    
    # Retrieval parameters
    "top_k_default": 5,
    "fine_grained_top_k": 3,
}

# ==================== Knowledge-base build configuration ====================
KB_CONFIG = {
    "max_main_content_pages": 20,
    "min_text_length": 300,
    "batch_size": 2,
    "request_delay": 1.0,  # API request delay in seconds
    
    # Chunking configuration
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "min_chunk_length": 50,
}

# ==================== SentenceTransformer model management ====================
_sentence_transformer_model = None
_sentence_transformer_model_name = MODEL_CONFIG["sentence_transformer"]["model_name"]


def _looks_like_git_lfs_pointer(path: Path) -> bool:
    """Best-effort detection for a Git LFS pointer file."""
    try:
        if not path.exists() or path.stat().st_size > 1024:
            return False
        header = path.read_text(encoding="utf-8", errors="ignore")
        return header.startswith("version https://git-lfs.github.com/spec/v1")
    except Exception:
        return False

def set_sentence_transformer_model(model_name: str):
    """
    Set the Sentence Transformer model name to use
    
    Args:
        model_name: Model name or path
    """
    global _sentence_transformer_model_name, _sentence_transformer_model
    _sentence_transformer_model_name = model_name
    _sentence_transformer_model = None
    logger.info(f"Sentence Transformer model set to: {model_name}")

def get_sentence_transformer_model(force_reload: bool = False):
    """
    Get the global SentenceTransformer model
    
    Args:
        force_reload: Whether to force model reload
        
    Returns:
        SentenceTransformer model instance
    """
    global _sentence_transformer_model, _sentence_transformer_model_name
    
    if _sentence_transformer_model is not None and not force_reload:
        return _sentence_transformer_model
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Resolve model path
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        local_model_dir = Path(MODEL_CONFIG["sentence_transformer"]["local_path"])
        hf_fallback_model = _sentence_transformer_model_name
        
        # Priority order:
        # 1. Explicit local path
        # 2. Local model directory from config
        # 3. HuggingFace repo id
        model_path = _sentence_transformer_model_name
        if os.path.exists(model_path):
            pass
        elif local_model_dir.exists():
            model_path = str(local_model_dir)
        else:
            logger.info(
                "Local SentenceTransformer directory is absent. Falling back to remote model '%s'.",
                hf_fallback_model,
            )
            model_path = hf_fallback_model

        logger.info(f"Loading SentenceTransformer model from: {model_path}")
        
        # Validate that model files exist
        if os.path.exists(model_path):
            config_exists = os.path.exists(os.path.join(model_path, "config.json"))
            weights_path = Path(model_path) / "pytorch_model.bin"
            safetensors_path = Path(model_path) / "model.safetensors"
            has_weights = weights_path.exists() or safetensors_path.exists()
            missing_files = []
            if not config_exists:
                missing_files.append("config.json")
            if not has_weights:
                missing_files.append("model.safetensors|pytorch_model.bin")
            if missing_files:
                logger.warning(f"Model files missing: {missing_files}, will try to download if needed")
                model_path = hf_fallback_model
            elif _looks_like_git_lfs_pointer(weights_path) and not safetensors_path.exists():
                logger.warning(
                    "Local model weights at %s are a Git LFS pointer and no local safetensors file is present. Falling back to remote model '%s'.",
                    weights_path,
                    hf_fallback_model,
                )
                model_path = hf_fallback_model
        
        # Load model
        model_kwargs = {"weights_only": False}

        try:
            _sentence_transformer_model = SentenceTransformer(
                model_path,
                local_files_only=True,
                model_kwargs=model_kwargs,
            )
            logger.info("Model loaded successfully with local_files_only=True")
        except Exception as e:
            logger.warning(f"Failed to load with local_files_only=True: {e}")
            logger.info("Attempting to load without local_files_only...")
            _sentence_transformer_model = SentenceTransformer(
                model_path,
                model_kwargs=model_kwargs,
            )
        
        return _sentence_transformer_model
        
    except ImportError:
        logger.error("sentence-transformers not installed. Please install it to use dense retrieval.")
        raise ImportError("sentence-transformers not installed")

def clear_model_cache():
    """Clear model cache"""
    global _sentence_transformer_model
    _sentence_transformer_model = None
    logger.info("Sentence Transformer model cache cleared")

# ==================== spaCy model management ====================
_spacy_model = None
_spacy_available = False

def get_spacy_model(force_reload: bool = False):
    """
    Get the global spaCy model
    
    Args:
        force_reload: Whether to force model reload
        
    Returns:
        spaCy model instance or None if unavailable
    """
    global _spacy_model, _spacy_available
    
    if _spacy_model is not None and not force_reload:
        return _spacy_model
    
    try:
        import spacy
        model_name = MODEL_CONFIG["spacy"]["model_name"]
        disable = MODEL_CONFIG["spacy"]["disable_components"]
        
        logger.info(f"Loading spaCy model: {model_name}")
        
        # RAG helper comment.
        _spacy_model = spacy.load(model_name, disable=disable)
        
        # Ensure a sentence segmenter exists
        if 'sentencizer' not in _spacy_model.pipe_names:
            _spacy_model.add_pipe('sentencizer')
            logger.info("Added sentencizer to spaCy pipeline")
        
        _spacy_available = True
        logger.info("spaCy model loaded successfully")
        return _spacy_model
    except ImportError:
        logger.warning("spaCy not installed. Please install it to use better sentence splitting.")
        _spacy_available = False
        return None
    except OSError as e:
        logger.warning(f"spaCy model not found: {e}")
        _spacy_available = False
        return None

def is_spacy_available() -> bool:
    """Check whether spaCy is available"""
    global _spacy_available
    if _spacy_model is None:
        get_spacy_model()
    return _spacy_available

def clear_spacy_model():
    """Clear spaCy model cache"""
    global _spacy_model, _spacy_available
    _spacy_model = None
    _spacy_available = False
    logger.info("spaCy model cache cleared")

# ==================== Utility functions ====================
def ensure_dir(path: Path) -> Path:
    """Ensure directories exist"""
    path.mkdir(parents=True, exist_ok=True)
    return path

def get_path(key: str) -> Path:
    """Get a predefined path"""
    if key in DIRS:
        return DIRS[key]
    elif key in FILES:
        return FILES[key]
    else:
        raise KeyError(f"Unknown path key: {key}")

def get_model_path() -> Path:
    """Resolve model path"""
    return MODEL_CONFIG["sentence_transformer"]["local_path"]

# RAG helper comment.
def call_llm(
    llm_client,
    user_prompt: str,
    system_prompt: str = None,
    temperature: float = 0.3,
    *,
    agent: str = "rag",
) -> str:
    """
    Convenience function for calling an LLM API
    
    Args:
        llm_client: LLM client instance
        user_prompt: User prompt
        system_prompt: System prompt
        temperature: Temperature parameter
        
    Returns:
        LLM response text
    """
    if llm_client is None:
        raise ValueError("LLM client was not provided")
    
    # Adapt different client types
    if hasattr(llm_client, 'invoke'):
        # ChatOpenAI object
        from langchain_core.messages import SystemMessage, HumanMessage
        langchain_messages = []
        if system_prompt:
            langchain_messages.append(SystemMessage(content=system_prompt))
        langchain_messages.append(HumanMessage(content=user_prompt))

        old_session_id = None
        session_changed = False
        if hasattr(llm_client, "session_id") and str(agent or "").strip() != "runtime_v2":
            old_session_id = getattr(llm_client, "session_id", None)
            base_session = str(old_session_id or "cytobridge-internal").strip()
            digest = hashlib.sha1(f"{agent}\n{user_prompt}".encode("utf-8", errors="ignore")).hexdigest()[:10]
            internal_session_id = derive_bounded_stateful_session_id(
                base_session,
                f"-internal-{str(agent or 'rag').strip() or 'rag'}-{digest}",
            )
            try:
                setattr(llm_client, "session_id", internal_session_id)
                session_changed = True
            except Exception:
                session_changed = False

        try:
            response = invoke_with_retry(
                llm_client,
                langchain_messages,
                logger=logger,
                agent=agent,
            )
        finally:
            if session_changed:
                try:
                    setattr(llm_client, "session_id", old_session_id)
                except Exception:
                    logger.debug("Failed to restore stateful LLM session id after internal RAG call", exc_info=True)
        if hasattr(response, 'content'):
            return response.content
        return str(response)
    else:
        # Assume a custom LLMClient object
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        
        response = llm_client.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=2000
        )
    
    if not response:
        raise ValueError("LLM call failed")
    
    return response

# ==================== Environment checks ====================
def check_environment() -> Dict[str, Any]:
    """Check environment configuration"""
    status = {
        "paths": {},
        "files": {},
        "config": {},
        "model": {},
        "spacy": {}
    }
    
    # Check directories
    for name, path in DIRS.items():
        status["paths"][name] = {
            "exists": path.exists(),
            "path": str(path)
        }
    
    # Check files
    for name, path in FILES.items():
        status["files"][name] = {
            "exists": path.exists(),
            "path": str(path)
        }
    
    # Check SentenceTransformer model
    model_path = get_model_path()
    status["model"] = {
        "name": MODEL_CONFIG["sentence_transformer"]["model_name"],
        "local_path": str(model_path),
        "exists": model_path.exists(),
        "config_exists": (model_path / "config.json").exists() if model_path.exists() else False,
        "model_exists": (model_path / "pytorch_model.bin").exists() if model_path.exists() else False,
    }
    
    # Check spaCy
    try:
        import spacy
        status["spacy"]["installed"] = True
        try:
            spacy.load(MODEL_CONFIG["spacy"]["model_name"])
            status["spacy"]["model_available"] = True
        except:
            status["spacy"]["model_available"] = False
            status["spacy"]["model_name"] = MODEL_CONFIG["spacy"]["model_name"]
    except ImportError:
        status["spacy"]["installed"] = False
        status["spacy"]["model_available"] = False
    
    return status

# Export common functions and variables
__all__ = [
    'PROJECT_ROOT',
    'RAG_DIR',
    'DIRS',
    'FILES',
    'MODEL_CONFIG',
    'SEARCH_CONFIG',
    'KB_CONFIG',
    'ensure_dir',
    'get_path',
    'get_model_path',
    'check_environment',
    'set_sentence_transformer_model',
    'get_sentence_transformer_model',
    'clear_model_cache',
    'get_spacy_model',
    'is_spacy_available',
    'clear_spacy_model',
    'call_llm',
]
