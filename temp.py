"""Script to manually trigger initial training for untrained processors."""

import asyncio
import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from document_ai import (
    ProcessedDocument,
    IncrementalTrainingBatch,
    AutomatedTrainingConfig,
    IncrementalTrainingManager,
    DocumentType,
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


async def trigger_training():
    """Manually trigger training for the processor."""
    processor_id = os.getenv("DOCUMENT_AI_PROCESSOR_ID")
    
    if not processor_id:
        logger.error("DOCUMENT_AI_PROCESSOR_ID not set")
        return
    
    # Cleanup: Delete all ProcessedDocument records for this processor to reset counts
    await ProcessedDocument.find({"processor_id": processor_id}).delete()
    
    # Check current document status
    pending_count = await ProcessedDocument.find({
        "processor_id": processor_id,
        "status": "pending"
    }).count()
    
    completed_count = await ProcessedDocument.find({
        "processor_id": processor_id,
        "status": "completed"
    }).count()
    
    total_count = await ProcessedDocument.find({
        "processor_id": processor_id
    }).count()
    
    logger.info(f"\nDocument Status for processor {processor_id}:")
    logger.info(f"  Total documents: {total_count}")
    logger.info(f"  Pending (for initial training): {pending_count}")
    logger.info(f"  Completed (for retraining): {completed_count}")
    
    # Update or create training config with lower threshold for testing
    config = await AutomatedTrainingConfig.find_one({"processor_id": processor_id})
    if not config:
        config = AutomatedTrainingConfig(
            processor_id=processor_id,
            enabled=True,
            check_interval_minutes=60,
            min_documents_for_training=2,  # Lower threshold for testing
            min_accuracy_for_deployment=0.7,
            document_types=list(DocumentType),
        )
        await config.save()
        logger.info("Created training config with lower threshold for testing")
    else:
        # Update threshold for testing
        config.min_documents_for_training = 2
        await config.save()
        logger.info("Updated training config with lower threshold for testing")
    
    # Initialize training manager
    manager = IncrementalTrainingManager()
    
    # Trigger training
    logger.info("\nTriggering training...")
    batch = await manager.run_incremental_training(processor_id)
    
    if batch:
        logger.info(f"\n✅ Training triggered successfully!")
        logger.info(f"  Batch ID: {batch.batch_id}")
        logger.info(f"  Documents: {len(batch.document_ids)}")
        logger.info(f"  Status: {batch.status}")
        logger.info("\nMonitor the training progress in the Google Cloud Console")
    else:
        logger.info("\n❌ Training was not triggered")
        logger.info("Possible reasons:")
        logger.info("- Not enough documents")
        logger.info("- Training already in progress")
        logger.info("- Training disabled in config")


async def main():
    """Main function."""
    if not await init_db():
        return
    
    await trigger_training()


if __name__ == "__main__":
    asyncio.run(main())