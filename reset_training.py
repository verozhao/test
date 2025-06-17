"""Script to reset training state."""

import asyncio
import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from document_ai import (
    ProcessedDocument,
    IncrementalTrainingBatch,
    AutomatedTrainingConfig,
    DocumentAIStatus,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def init_db():
    """Initialize MongoDB connection."""
    try:
        client = AsyncIOMotorClient("mongodb://localhost:27017")
        await init_beanie(
            database=client.document_ai,
            document_models=[ProcessedDocument, IncrementalTrainingBatch, AutomatedTrainingConfig]
        )
        logger.info("Database initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

async def reset_training_state():
    """Reset the training state."""
    try:
        # Delete all training batches
        result = await IncrementalTrainingBatch.find().delete()
        logger.info(f"Deleted {result} training batches")
        
        # Reset all documents to PENDING state
        result = await ProcessedDocument.find().update({
            "$set": {
                "status": DocumentAIStatus.PENDING,
                "used_for_training": False,
                "training_batch_id": None
            }
        })
        logger.info(f"Reset {result} documents to PENDING state")
        
        # Update training config to use lower threshold for testing
        config = await AutomatedTrainingConfig.find_one()
        if config:
            config.min_documents_for_training = 2  # Lower threshold for testing
            await config.save()
            logger.info("Updated training config with lower threshold")
        
    except Exception as e:
        logger.error(f"Error resetting training state: {str(e)}")

async def main():
    """Main function."""
    if not await init_db():
        return
    
    await reset_training_state()

if __name__ == "__main__":
    asyncio.run(main()) 