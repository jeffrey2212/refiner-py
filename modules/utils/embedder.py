"""Custom embedder implementation that uses local model cache"""
import os
import logging
from pathlib import Path
from fastembed import TextEmbedding
from agno.embedder.fastembed import FastEmbedEmbedder

logger = logging.getLogger(__name__)

class LocalFastEmbedder(FastEmbedEmbedder):
    """FastEmbed embedder that uses local model cache"""
    
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        """Initialize embedder with model name"""
        self.id = model_name
        self._model = TextEmbedding(
            model_name=model_name,
            max_length=512,
            cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "fastembed")
        )

    def get_embedding(self, text):
        """Get embeddings for text using model"""
        embeddings = list(self._model.embed([text]))  # embed expects a list
        return embeddings[0]  # return first embedding
