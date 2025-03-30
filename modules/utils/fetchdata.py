"""Module for fetching and processing data from Civitai API and storing in Qdrant"""
import os
import json
import requests
import streamlit as st
import argparse
from typing import List, Dict, Optional, Any
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.http import models
from agno.embedder.fastembed import FastEmbedEmbedder
import numpy as np
import logging
import time
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_MODELS = ["Illustrious", "Flux.1 D", "Pony"]
BATCH_SIZE = 200
REQUIRED_SECRETS = ["CIVITAI_API_KEY", "QDRANT_URL", "QDRANT_API_KEY", "COLLECTION_NAME"]

class CivitaiDataFetcher:
    def __init__(self, use_env: bool = False):
        """Initialize the data fetcher with API settings and Qdrant client"""
        logger.info("Initializing CivitaiDataFetcher...")
        
        if use_env:
            # Try to load from .env file
            if not load_dotenv():
                logger.warning("No .env file found")
            
            # Get secrets from environment variables
            secrets = {}
            for secret in REQUIRED_SECRETS:
                secrets[secret] = os.getenv(secret)
        else:
            # Get secrets from streamlit
            secrets = st.secrets

        # Validate required secrets
        missing_secrets = []
        for secret in REQUIRED_SECRETS:
            if secret not in secrets or not secrets[secret]:
                missing_secrets.append(secret)
        
        if missing_secrets:
            raise ValueError(f"Missing required secrets: {', '.join(missing_secrets)}")
        
        self.api_key = secrets["CIVITAI_API_KEY"]
        self.base_url = "https://civitai.com/api/v1/images"
        self.embedder = FastEmbedEmbedder()
        self.qdrant_client = QdrantClient(
            url=secrets["QDRANT_URL"],
            api_key=secrets["QDRANT_API_KEY"]
        )
        self.collection_name = secrets["COLLECTION_NAME"]
        self.temp_file = "modules/utils/tempfile.json"
        logger.info(f"Initialized with collection: {self.collection_name}")
        
        # Initialize temp file if it doesn't exist
        if not os.path.exists(self.temp_file):
            self._save_cursor('', 0)

    def _get_last_cursor(self) -> Optional[str]:
        """Get the last cursor from temp file"""
        try:
            if os.path.exists(self.temp_file):
                with open(self.temp_file, 'r') as f:
                    data = json.load(f)
                cursor = data.get('cursor', '')
                logger.info(f"Retrieved last cursor: {cursor}")
                return cursor
            logger.info("No previous cursor found")
            return None
        except Exception as e:
            logger.error(f"Error reading cursor file: {e}")
            return None

    def _save_cursor(self, cursor: str, total_processed: int) -> None:
        """Save cursor and total processed count to temp file"""
        try:
            data = {
                'cursor': cursor,
                'total_processed': total_processed
            }
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.temp_file), exist_ok=True)
            
            with open(self.temp_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.debug(f"Saved cursor: {cursor} and total processed: {total_processed}")
        except Exception as e:
            logger.error(f"Error saving cursor: {str(e)}")

    def _get_saved_state(self) -> tuple[str, int]:
        """Get saved cursor and total processed count from temp file"""
        try:
            if os.path.exists(self.temp_file):
                with open(self.temp_file, 'r') as f:
                    try:
                        data = json.load(f)
                        cursor = str(data.get('cursor', ''))
                        total_processed = int(data.get('total_processed', 0))
                        logger.info(f"Retrieved last cursor: {cursor}, total processed: {total_processed}")
                        return cursor, total_processed
                    except json.JSONDecodeError as e:
                        logger.error(f"Invalid JSON in temp file: {str(e)}")
                        # Reset the file with default values
                        self._save_cursor('', 0)
        except Exception as e:
            logger.error(f"Error reading cursor: {str(e)}")
        return '', 0

    def _process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single item from the API response"""
        try:
            if not isinstance(item, dict):
                logger.error(f"Invalid item format: {type(item)}")
                return None

            # Extract and validate ID
            item_id = item.get('id')
            if not item_id:
                logger.debug(f"Item has no ID")
                return None
            try:
                # Convert ID to integer for Qdrant
                item_id = int(str(item_id))
            except (ValueError, TypeError):
                logger.error(f"Invalid ID format: {item_id}")
                return None

            # Extract and validate base model
            base_model = item.get('baseModel')
            if not base_model:
                logger.debug(f"Item {item_id} has no base model")
                return None
            if base_model not in ALLOWED_MODELS:
                logger.debug(f"Item {item_id} has unsupported model: {base_model}")
                return None

            # Extract and validate meta
            meta = item.get('meta')
            if not isinstance(meta, dict):
                logger.debug(f"Item {item_id} has invalid meta format: {type(meta)}")
                return None

            # Extract and validate prompt
            prompt = meta.get('prompt')
            if not prompt:
                logger.debug(f"Item {item_id} has no prompt")
                return None

            # Create processed record
            record = {
                'id': item_id,  # Store as integer
                'url': item.get('url', ''),
                'prompt': prompt,
                'negative_prompt': meta.get('negativePrompt', ''),
                'model_name': base_model,
                'created_at': item.get('createdAt', ''),
                'meta': {
                    'seed': meta.get('seed'),
                    'steps': meta.get('steps'),
                    'cfg_scale': meta.get('cfgScale'),
                    'sampler': meta.get('sampler'),
                    'width': meta.get('width'),
                    'height': meta.get('height')
                }
            }

            return record
        except Exception as e:
            logger.error(f"Error processing item: {str(e)}")
            logger.debug(f"Problematic item: {item}")
            return None

    def _check_duplicate(self, item_id: int) -> bool:
        """Check if an item already exists in Qdrant"""
        try:
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="id",
                            match=models.MatchValue(value=item_id)
                        )
                    ]
                ),
                limit=1
            )
            is_duplicate = len(results[0]) > 0
            if is_duplicate:
                logger.debug(f"Item {item_id} already exists in Qdrant")
            return is_duplicate
        except Exception as e:
            logger.error(f"Error checking duplicate for {item_id}: {e}")
            return False

    def _store_in_qdrant(self, records: List[Dict[str, Any]]) -> None:
        """Store processed records in Qdrant"""
        try:
            logger.info(f"Processing {len(records)} records for storage...")
            points = []
            skipped = 0
            for record in records:
                if self._check_duplicate(record['id']):
                    skipped += 1
                    continue

                # Generate embeddings for the prompt
                logger.debug(f"Generating embeddings for item {record['id']}")
                embeddings = self.embedder.get_embedding(record['prompt'])
                
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
                    logger.error(f"Invalid embedding format for record {record['id']}: {type(embeddings)}")
                    continue
                
                try:
                    # Convert any numpy numbers to native Python types
                    embeddings = [float(x) for x in embeddings]
                except (TypeError, ValueError) as e:
                    logger.error(f"Error converting embeddings to float for record {record['id']}: {e}")
                    continue

                # Create point with validated embeddings
                try:
                    point = models.PointStruct(
                        id=record['id'],  # Already an integer from _process_item
                        vector=embeddings,
                        payload=record
                    )
                    points.append(point)
                except Exception as e:
                    logger.error(f"Error creating point struct for record {record['id']}: {e}")
                    logger.debug(f"Point data - ID type: {type(record['id'])}, Vector length: {len(embeddings)}")
                    continue

            if points:
                logger.info(f"Storing {len(points)} new points in Qdrant (skipped {skipped} duplicates)")
                try:
                    # Log first point for debugging
                    sample_point = points[0]
                    logger.debug(f"Sample point - ID: {sample_point.id} (type: {type(sample_point.id)})")
                    logger.debug(f"Sample point - Vector type: {type(sample_point.vector)}, length: {len(sample_point.vector)}")
                    logger.debug(f"Sample point - First few vector values: {sample_point.vector[:5]}")
                    
                    # Store points in batches to avoid timeouts
                    batch_size = 50
                    for i in range(0, len(points), batch_size):
                        batch = points[i:i + batch_size]
                        logger.info(f"Storing batch {i//batch_size + 1} of {(len(points)-1)//batch_size + 1} ({len(batch)} points)")
                        self.qdrant_client.upsert(
                            collection_name=self.collection_name,
                            points=batch,
                            wait=True
                        )
                        time.sleep(0.5)  # Small delay between batches
                    
                    logger.info("Storage complete")
                except Exception as e:
                    logger.error(f"Error during Qdrant upsert: {str(e)}")
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                        logger.error(f"Qdrant response: {e.response.text}")
                    raise
            else:
                logger.info(f"No new points to store (skipped {skipped} duplicates)")

        except Exception as e:
            logger.error(f"Error storing in Qdrant: {str(e)}")
            raise

    def fetch_and_store(self, target_count: int = 1000) -> None:
        """Fetch data from Civitai API and store in Qdrant until target count is reached"""
        logger.info(f"Starting fetch and store process with target count: {target_count}")
        
        # Get saved state
        cursor, prev_processed = self._get_saved_state()
        logger.info(f"Previous run processed {prev_processed} records")
        
        # Reset total_processed for new run but keep the cursor
        total_processed = 0
        batch = 1

        while total_processed < target_count:
            batch_size = min(BATCH_SIZE, target_count - total_processed)
            logger.info(f"Fetching batch {batch} (size: {batch_size}, total processed: {total_processed})")

            if cursor:
                logger.info(f"Using cursor: {cursor}")
            else:
                logger.info("Starting from beginning (no cursor)")

            # Fetch data from API
            logger.info("Fetching data from Civitai API...")
            response = requests.get(
                self.base_url,
                params={
                    'limit': batch_size,
                    'sort': 'Most Reactions',
                    'period': 'Month',
                    'cursor': cursor
                },
                headers={'Authorization': f'Bearer {self.api_key}'}
            )
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, dict) or 'items' not in data:
                logger.error(f"Invalid API response format: {data}")
                break

            items = data.get('items', [])
            if not items:
                logger.warning("No items received from API")
                break

            logger.info(f"Received {len(items)} items from API")

            # Process items
            processed_records = []
            for item in items:
                record = self._process_item(item)
                if record:
                    processed_records.append(record)

            logger.info(f"Processed {len(processed_records)} valid records (skipped {len(items) - len(processed_records)})")

            if processed_records:
                # Store in Qdrant
                try:
                    self._store_in_qdrant(processed_records)
                    total_processed += len(processed_records)
                    # Save cursor and cumulative total
                    self._save_cursor(data.get('metadata', {}).get('nextCursor', ''), prev_processed + total_processed)
                except Exception as e:
                    logger.error(f"Error storing batch: {e}")
                    break

            # Update cursor for next batch
            cursor = data.get('metadata', {}).get('nextCursor')
            if not cursor:
                logger.info("No more data available (no next cursor)")
                break

            batch += 1
            time.sleep(1)  # Rate limiting

        logger.info(f"Finished processing. New records: {total_processed}, Total records: {prev_processed + total_processed}")

    def get_stored_data(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve stored data from Qdrant"""
        try:
            logger.info(f"Retrieving {limit} records from Qdrant...")
            results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=limit,
                with_payload=True,
                with_vectors=False
            )
            records = [point.payload for point in results[0]]
            logger.info(f"Retrieved {len(records)} records")
            return records
        except Exception as e:
            logger.error(f"Error retrieving data: {e}")
            return []

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fetch and store data from Civitai API')
    parser.add_argument('--target', type=int, default=1000,
                      help='Target number of records to fetch (default: 1000)')
    parser.add_argument('--use-env', action='store_true',
                      help='Use .env file for secrets instead of streamlit secrets')
    
    args = parser.parse_args()
    
    logger.info(f"Starting Civitai data fetcher with target count: {args.target}")
    fetcher = CivitaiDataFetcher(use_env=args.use_env)
    fetcher.fetch_and_store(args.target)
    logger.info("Process complete")