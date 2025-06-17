"""Script to check training batch status."""

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

async def check_training_status():
    """Check the status of all training batches."""
    try:
        # Get all training batches
        batches = await IncrementalTrainingBatch.find().to_list()
        
        if not batches:
            logger.info("No training batches found")
            return
        
        logger.info(f"\nFound {len(batches)} training batches:")
        
        for batch in batches:
            logger.info(f"\nBatch: {batch.batch_id}")
            logger.info(f"  Status: {batch.status}")
            logger.info(f"  Started at: {batch.started_at}")
            logger.info(f"  Completed at: {batch.completed_at if hasattr(batch, 'completed_at') else 'Not completed'}")
            logger.info(f"  Deployed at: {batch.deployed_at if hasattr(batch, 'deployed_at') else 'Not deployed'}")
            logger.info(f"  Error: {batch.error_message if hasattr(batch, 'error_message') else 'No error'}")
            logger.info(f"  Document count: {len(batch.document_ids)}")
            logger.info(f"  Document types: {batch.document_type_counts}")
        
        # Check pending documents
        pending_docs = await ProcessedDocument.find({
            "status": DocumentAIStatus.PENDING,
        }).count()
        
        completed_docs = await ProcessedDocument.find({
            "status": DocumentAIStatus.COMPLETED,
        }).count()
        
        logger.info(f"\nDocument Status:")
        logger.info(f"  Pending documents: {pending_docs}")
        logger.info(f"  Completed documents: {completed_docs}")
        
    except Exception as e:
        logger.error(f"Error checking training status: {str(e)}")

async def main():
    """Main function."""
    if not await init_db():
        return
    
    await check_training_status()

if __name__ == "__main__":
    asyncio.run(main()) 