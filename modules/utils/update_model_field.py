"""Script to update model_name to baseModel in existing Qdrant records"""
import os
import sys
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import numpy as np

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

from modules.utils.embedder import LocalFastEmbedder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def update_model_field():
    """Update model_name to baseModel in all records"""
    # Load environment variables
    if not load_dotenv():
        logger.warning("No .env file found")

    # Initialize Qdrant client
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = os.getenv("COLLECTION_NAME")

    if not all([qdrant_url, qdrant_api_key, collection_name]):
        raise ValueError("Missing required environment variables")

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    embedder = LocalFastEmbedder()
    
    try:
        # Get all records that have model_name field
        batch_size = 100
        offset = None
        total_updated = 0
        
        while True:
            # Get all points in batches
            results = client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=True,
                limit=batch_size,
                offset=offset
            )
            
            records, next_offset = results
            
            if not records:
                break
                
            logger.info(f"Processing batch of {len(records)} records")
            
            # Update each record
            points_to_update = []
            for record in records:
                payload = record.payload
                if 'model_name' in payload:
                    # Create new payload with baseModel
                    new_payload = dict(payload)
                    new_payload['baseModel'] = new_payload.pop('model_name')
                    
                    # Re-generate embeddings for the prompt
                    prompt = new_payload.get('prompt', '')
                    if not prompt:
                        logger.warning(f"Record {record.id} has no prompt, skipping")
                        continue
                        
                    embeddings = embedder.get_embedding(prompt)
                    
                    # Handle different embedding formats
                    if isinstance(embeddings, list) and len(embeddings) > 0:
                        if isinstance(embeddings[0], np.ndarray):
                            embeddings = embeddings[0].tolist()
                        else:
                            embeddings = embeddings[0]  # Already a list
                    elif isinstance(embeddings, np.ndarray):
                        embeddings = embeddings.tolist()
                    
                    # Validate embedding format
                    if not isinstance(embeddings, list):
                        logger.error(f"Invalid embedding format for record {record.id}: {type(embeddings)}")
                        continue
                    
                    try:
                        # Convert any numpy numbers to native Python types
                        embeddings = [float(x) for x in embeddings]
                    except (TypeError, ValueError) as e:
                        logger.error(f"Error converting embeddings to float for record {record.id}: {e}")
                        continue
                    
                    # Create point for update
                    points_to_update.append(models.PointStruct(
                        id=record.id,
                        payload=new_payload,
                        vector=embeddings
                    ))
            
            if points_to_update:
                # Update records in batch
                client.upsert(
                    collection_name=collection_name,
                    points=points_to_update,
                    wait=True
                )
                total_updated += len(points_to_update)
                logger.info(f"Updated {len(points_to_update)} records in this batch")
            
            if not next_offset:
                break
                
            offset = next_offset
            
        logger.info(f"Update complete. Total records updated: {total_updated}")
        
    except Exception as e:
        logger.error(f"Error updating records: {str(e)}")
        raise

if __name__ == "__main__":
    update_model_field()
