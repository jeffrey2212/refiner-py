"""Script to move prompt and negative prompt fields from meta to root level in Qdrant records"""
import os
import sys
import logging
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def update_prompt_fields():
    """Move prompt and negative prompt fields from meta to root level in all records"""
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
    
    try:
        # Get all records in batches
        batch_size = 100
        offset = None
        total_updated = 0
        
        while True:
            # Get points in batches
            results = client.scroll(
                collection_name=collection_name,
                with_payload=True,
                with_vectors=True,  # Changed to True to get vectors
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
                meta = payload.get('meta', {})
                
                # Check if prompt fields exist in meta and not in root
                if ('prompt' not in payload and 'prompt' in meta) or \
                   ('negativePrompt' not in payload and 'negativePrompt' in meta):
                    # Create new payload
                    new_payload = dict(payload)
                    new_meta = dict(meta)
                    
                    # Move prompt fields to root level if they exist in meta
                    if 'prompt' in meta:
                        new_payload['prompt'] = new_meta.pop('prompt')
                    if 'negativePrompt' in meta:
                        new_payload['negativePrompt'] = new_meta.pop('negativePrompt')
                    
                    # Update meta in payload
                    new_payload['meta'] = new_meta
                    
                    # Add to points to update with existing vector
                    points_to_update.append(models.PointStruct(
                        id=record.id,
                        payload=new_payload,
                        vector=record.vector
                    ))
            
            # Update points in batch if any
            if points_to_update:
                client.upsert(
                    collection_name=collection_name,
                    points=points_to_update
                )
                total_updated += len(points_to_update)
                logger.info(f"Updated {len(points_to_update)} records in this batch")
            
            # Update offset for next batch
            offset = next_offset
            if not offset:
                break
        
        logger.info(f"Total records updated: {total_updated}")
                
    except Exception as e:
        logger.error(f"Error updating records: {e}")
        raise

if __name__ == "__main__":
    update_prompt_fields()
