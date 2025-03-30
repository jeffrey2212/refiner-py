"""Script to pre-download the FastEmbed model"""
import logging
from agno.embedder.fastembed import FastEmbedEmbedder

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_model():
    """Download the FastEmbed model"""
    logger.info("Initializing embedder...")
    embedder = FastEmbedEmbedder()
    
    # Test the embedder with a simple prompt to trigger download
    logger.info("Downloading model by testing with a sample prompt...")
    embeddings = embedder.get_embedding("test prompt")
    logger.info("Model downloaded and tested successfully!")

if __name__ == "__main__":
    download_model()
