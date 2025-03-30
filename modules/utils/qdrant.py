"""Vector database operations for prompt similarity search using Qdrant"""
import streamlit as st
from typing import List, Optional, Tuple
from agno.vectordb.qdrant import Qdrant
from agno.embedder.fastembed import FastEmbedEmbedder
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
import numpy as np
import json
import traceback

def get_qdrant_client() -> Optional[Tuple[Qdrant, QdrantClient]]:
    """Get a connection to the Qdrant database with FastEmbed"""
    try:
        qdrant_api_key = st.secrets["QDRANT_API_KEY"]
        if not qdrant_api_key:
            raise ValueError("QDRANT_API_KEY not found in secrets.toml")
        
        qdrant_url = st.secrets["QDRANT_URL"]
        if not qdrant_url:
            raise ValueError("QDRANT_URL not properly configured in secrets.toml")
        
        collection_name = st.secrets["COLLECTION_NAME"]
        print(f"Connecting to collection: {collection_name}")
        
        # Create both Agno wrapper and direct client
        embedder = FastEmbedEmbedder()
        agno_client = Qdrant(
            collection=collection_name,
            url=qdrant_url,
            api_key=qdrant_api_key,
            embedder=embedder
        )
        
        # Create direct Qdrant client
        qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key
        )
        
        return agno_client, qdrant_client
        
    except Exception as e:
        print(f"Error connecting to Qdrant: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        return None

def get_similar_prompts(prompt: str, model_name: str = "general", k: int = 5) -> List[str]:
    """Get similar prompts from Qdrant"""
    try:
        clients = get_qdrant_client()
        if not clients:
            return []
            
        agno_client, qdrant_client = clients
            
        # Get embeddings for the query and convert to list
        print(f"Getting embeddings for prompt: {prompt[:50]}...")
        embeddings = agno_client.embedder.get_embedding(prompt)
        
        # Handle different embedding formats
        if isinstance(embeddings, list) and len(embeddings) > 0 and isinstance(embeddings[0], np.ndarray):
            # If it's a list of numpy arrays, take the first one and convert to list
            embeddings = embeddings[0].tolist()
        elif isinstance(embeddings, np.ndarray):
            # If it's a single numpy array, convert to list
            embeddings = embeddings.tolist()
            
        print(f"Embedding shape/length: {len(embeddings)}")
        print(f"Embedding type: {type(embeddings)}, First few values: {embeddings[:3]}")
        
        # Use direct Qdrant client for search
        collection_name = agno_client.collection.replace("-", "_")
        print(f"Searching in collection: {collection_name}")
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="model_name",
                    match=MatchValue(value=model_name)
                )
            ]
        ) if model_name != "general" else None
        
        results = qdrant_client.search(
            collection_name=collection_name,
            query_vector=embeddings,
            limit=k,
            query_filter=query_filter
        )
        
        # Extract prompts from results
        prompts = []
        for hit in results:
            if 'prompt' in hit.payload:
                prompts.append(hit.payload['prompt'])
        print(f"Found {len(prompts)} similar prompts")
                
        return prompts
        
    except Exception as e:
        print(f"Error getting similar prompts: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        return []

def format_prompt_for_rag(similar_prompts):
    """Format similar prompts for RAG context"""
    context = []
    for idx, p in enumerate(similar_prompts, 1):
        prompt_text = f"Example {idx}:\n"
        prompt_text += f"Prompt: {p}\n"
        prompt_text += f"Score: 0.0\n"
        context.append(prompt_text)
    
    formatted_context = "\n".join(context)
    print(f"\n[DEBUG] RAG Context:\n{formatted_context}\n")
    return formatted_context

def add_prompt_to_db(prompt, model_name, metadata=None):
    """This function is deprecated as we're using the existing civitai_images collection"""
    print("Warning: add_prompt_to_db is deprecated. Prompts are stored in civitai_images collection.")
