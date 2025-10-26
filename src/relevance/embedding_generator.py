"""
Embedding Generator for KGCompass Relevance Scoring
Generates semantic embeddings for code entities and problem descriptions
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Union
import hashlib
import pickle
import os
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available. Install with: pip install transformers torch")


class EmbeddingGenerator:
    """
    Generates semantic embeddings for code entities and problem descriptions
    Following KGCompass methodology using sentence-transformers
    """
    
    def __init__(self, 
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 cache_dir: str = "workspace/embeddings_cache",
                 device: str = "auto"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of the embedding model to use
            cache_dir: Directory to cache embeddings
            device: Device to run model on ('cpu', 'cuda', or 'auto')
        """
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.tokenizer = None
        
        # Set device
        if device == "auto":
            if TRANSFORMERS_AVAILABLE:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = "cpu"
        else:
            self.device = device
            
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                # Try sentence-transformers first (easier to use)
                try:
                    self.model = SentenceTransformer(self.model_name, device=self.device)
                    self.model_type = "sentence_transformers"
                    self.logger.info(f"Loaded {self.model_name} using sentence-transformers")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to load with sentence-transformers: {e}")
            
            if TRANSFORMERS_AVAILABLE:
                # Fallback to transformers
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModel.from_pretrained(self.model_name)
                self.model.to(self.device)
                self.model_type = "transformers"
                self.logger.info(f"Loaded {self.model_name} using transformers")
                return
                
        except Exception as e:
            self.logger.error(f"Failed to load embedding model {self.model_name}: {e}")
            
        # If we reach here, no ML libraries are available
        self.model = None
        self.model_type = None
        raise ImportError("No ML libraries available. Please install: pip install sentence-transformers transformers torch")
    
    def _get_cache_path(self, text: str) -> Path:
        """Get cache file path for text"""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        return self.cache_dir / f"{text_hash}.pkl"
    
    def _load_from_cache(self, text: str) -> Optional[np.ndarray]:
        """Load embedding from cache"""
        cache_path = self._get_cache_path(text)
        if cache_path.exists():
            try:
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cached embedding: {e}")
        return None
    
    def _save_to_cache(self, text: str, embedding: np.ndarray):
        """Save embedding to cache"""
        cache_path = self._get_cache_path(text)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding, f)
        except Exception as e:
            self.logger.warning(f"Failed to cache embedding: {e}")
    
    
    def generate_embedding(self, text: str, use_cache: bool = True) -> np.ndarray:
        """
        Generate embedding for text
        
        Args:
            text: Input text to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            Embedding vector as numpy array
        """
        if not text or not text.strip():
            return np.zeros(768)  # Return zero vector for empty text
        
        # Check cache first
        if use_cache:
            cached_embedding = self._load_from_cache(text)
            if cached_embedding is not None:
                return cached_embedding
        
        # Generate embedding
        embedding = self._generate_embedding_internal(text)
        
        # Cache the result
        if use_cache:
            self._save_to_cache(text, embedding)
            
        return embedding
    
    def _generate_embedding_internal(self, text: str) -> np.ndarray:
        """Internal method to generate embedding"""
        if self.model_type == "sentence_transformers":
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
            
        elif self.model_type == "transformers":
            # Tokenize and encode
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, 
                                  padding=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use mean pooling of last hidden states
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                return embedding.squeeze()
                
        else:
            raise RuntimeError("No embedding model available. This should not happen if initialization succeeded.")
    
    def generate_batch_embeddings(self, texts: List[str], use_cache: bool = True) -> List[np.ndarray]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            use_cache: Whether to use cached embeddings
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            if use_cache:
                cached_embedding = self._load_from_cache(text)
                if cached_embedding is not None:
                    embeddings.append(cached_embedding)
                    continue
            
            # Need to generate this embedding
            embeddings.append(None)  # Placeholder
            uncached_texts.append(text)
            uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            if self.model_type == "sentence_transformers":
                batch_embeddings = self.model.encode(uncached_texts, convert_to_numpy=True)
            else:
                # Generate one by one for other model types
                batch_embeddings = [self._generate_embedding_internal(text) for text in uncached_texts]
            
            # Fill in the results and cache them
            for i, embedding in enumerate(batch_embeddings):
                idx = uncached_indices[i]
                embeddings[idx] = embedding
                
                if use_cache:
                    self._save_to_cache(uncached_texts[i], embedding)
        
        return embeddings
    
    def compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score [0, 1]
        """
        if embedding1.size == 0 or embedding2.size == 0:
            return 0.0
            
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Ensure result is in [0, 1] range (convert from [-1, 1])
        return max(0.0, (similarity + 1.0) / 2.0)
    
    def prepare_code_entity_text(self, entity: Dict) -> str:
        """
        Prepare text representation of a code entity for embedding
        
        Args:
            entity: Code entity dictionary
            
        Returns:
            Text representation suitable for embedding
        """
        parts = []
        
        # Add entity type and name
        entity_type = entity.get('type', 'unknown')
        entity_name = entity.get('name', '')
        if entity_name:
            parts.append(f"{entity_type}: {entity_name}")
        
        # Add file path context
        file_path = entity.get('file_path', '')
        if file_path:
            parts.append(f"in file: {file_path}")
        
        # Add signature if available
        signature = entity.get('signature', '')
        if signature:
            parts.append(f"signature: {signature}")
        
        # Add docstring or comments if available
        docstring = entity.get('docstring', '')
        if docstring:
            parts.append(f"documentation: {docstring}")
        
        # Add code snippet if available (truncated)
        code = entity.get('code', '')
        if code:
            # Truncate long code snippets
            if len(code) > 500:
                code = code[:500] + "..."
            parts.append(f"code: {code}")
        
        return " | ".join(parts)
    
    def prepare_problem_description_text(self, problem_description: str) -> str:
        """
        Prepare problem description text for embedding
        
        Args:
            problem_description: Raw problem description
            
        Returns:
            Cleaned text suitable for embedding
        """
        # Basic text cleaning
        text = problem_description.strip()
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Truncate if too long (model-dependent, but 512 tokens is common limit)
        if len(text) > 2000:  # Rough character limit
            text = text[:2000] + "..."
        
        return text
    
    def clear_cache(self):
        """Clear the embedding cache"""
        try:
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
                self.logger.info("Embedding cache cleared")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.cache_dir.exists():
            return {"cached_embeddings": 0, "cache_size_mb": 0}
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            "cached_embeddings": len(cache_files),
            "cache_size_mb": total_size / (1024 * 1024)
        }
