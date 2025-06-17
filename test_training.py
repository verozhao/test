"""Test script for Document AI incremental training."""

import asyncio
import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from google.cloud import storage
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie

from document_ai import (
    EnhancedDocumentAIClient,
    DocumentType,
    ProcessedDocument,
    AutomatedTrainingConfig,
    IncrementalTrainingBatch,
    start_scheduler,
    stop_scheduler,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def upload_to_gcs(file_path: str, bucket_name: str) -> str:
    """Upload a file to GCS and return the GCS URI."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Use the filename as the blob name
    blob_name = os.path.basename(file_path)
    blob = bucket.blob(blob_name)
    
    # Upload the file
    blob.upload_from_filename(file_path)
    
    return f"gs://{bucket_name}/{blob_name}"

async def init_db():
    """Initialize MongoDB connection and Beanie."""
    try:
        # Connect to MongoDB
        client = AsyncIOMotorClient("mongodb://localhost:27017", serverSelectionTimeoutMS=2000)
        
        # Test connection
        await client.admin.command('ping')
        
        # Initialize Beanie with the document models
        await init_beanie(
            database=client.document_ai_test,
            document_models=[ProcessedDocument, AutomatedTrainingConfig, IncrementalTrainingBatch]
        )
        
        # Set up automated training config if it doesn't exist
        config = await AutomatedTrainingConfig.find_one({"processor_id": os.getenv("DOCUMENT_AI_PROCESSOR_ID")})
        if not config:
            config = AutomatedTrainingConfig(
                processor_id=os.getenv("DOCUMENT_AI_PROCESSOR_ID"),
                enabled=True,
                min_documents_for_training=2,  # Set to a small number for testing
                document_types=[doc_type for doc_type in DocumentType],
            )
            await config.save()
            logger.info("Created automated training configuration")
        else:
            logger.info(f"Found existing training config: {config}")
            
        return True
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False

async def process_documents(pdf_directory: str, bucket_name: str):
    """Process all PDF documents in the specified directory."""
    # Initialize client with GCS support and training enabled
    client = EnhancedDocumentAIClient(
        project_id=os.getenv("GCP_PROJECT_ID"),
        processor_id=os.getenv("DOCUMENT_AI_PROCESSOR_ID"),
        local_mode=False,  # Use GCS mode
        skip_db=False  # Enable database operations for training
    )
    
    # Get all PDF files in directory
    pdf_files = list(Path(pdf_directory).glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_directory}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each file
    for pdf_file in pdf_files:
        try:
            logger.info(f"Processing {pdf_file.name}...")
            
            # Upload to GCS first
            gcs_uri = upload_to_gcs(str(pdf_file), bucket_name)
            logger.info(f"Uploaded to GCS: {gcs_uri}")
            
            # Process the document
            result = await client.upload_and_process_document(
                document_path=gcs_uri,
                document_name=pdf_file.name,
                mime_type="application/pdf"
            )
            
            logger.info(f"Processing result for {pdf_file.name}:")
            logger.info(f"  Status: {result.status}")
            logger.info(f"  Document Type: {result.document_type}")
            logger.info(f"  Confidence: {getattr(result, 'confidence_score', None)}")
            if result.training_triggered:
                logger.info("  Training was triggered!")
            
        except Exception as e:
            logger.error(f"Error processing {pdf_file.name}: {str(e)}")

async def check_training_status():
    """Check the current training status."""
    try:
        # Get active training
        active_training = await IncrementalTrainingBatch.find_one({
            "processor_id": os.getenv("DOCUMENT_AI_PROCESSOR_ID"),
            "status": {"$in": ["pending", "training", "deploying"]}
        })
        
        if active_training:
            logger.info(f"Active training batch: {active_training.batch_id}")
            logger.info(f"  Status: {active_training.status}")
            logger.info(f"  Started at: {active_training.started_at}")
            logger.info(f"  Document count: {len(active_training.document_ids)}")
        
        # Get last completed training
        last_training = await IncrementalTrainingBatch.find_one({
            "processor_id": os.getenv("DOCUMENT_AI_PROCESSOR_ID"),
            "status": {"$in": ["deployed", "trained"]}
        }).sort([("completed_at", -1)])
        
        if last_training:
            logger.info(f"Last completed training: {last_training.batch_id}")
            logger.info(f"  Status: {last_training.status}")
            logger.info(f"  Completed at: {last_training.completed_at}")
            logger.info(f"  Accuracy: {last_training.accuracy_score}")
        
        # Count pending documents
        pending_docs = await ProcessedDocument.find({
            "processor_id": os.getenv("DOCUMENT_AI_PROCESSOR_ID"),
            "status": "completed",
            "used_for_training": False,
        }).count()
        
        logger.info(f"Pending documents for training: {pending_docs}")
        
    except Exception as e:
        logger.error(f"Error checking training status: {str(e)}")

async def main():
    """Main entry point."""
    # Get PDF directory and bucket name from environment or use defaults
    pdf_dir = os.getenv("PDF_DIRECTORY", "test_documents")
    bucket_name = os.getenv("GCS_BUCKET_NAME", "document-ai-test-bucket")
    
    if not os.path.exists(pdf_dir):
        logger.error(f"PDF directory {pdf_dir} does not exist")
        return
    
    # Initialize database
    if not await init_db():
        logger.error("Failed to initialize database. Exiting.")
        return
    
    # Start training scheduler
    logger.info("Starting training scheduler...")
    await start_scheduler()
    
    try:
        # Process documents
        logger.info("Processing documents...")
        await process_documents(pdf_dir, bucket_name)
        
        # Check initial training status
        logger.info("\nInitial training status:")
        await check_training_status()
        
        # Wait for training to complete
        logger.info("\nWaiting for training to complete (press Ctrl+C to stop)...")
        while True:
            await asyncio.sleep(60)  # Check every minute
            await check_training_status()
            
    except KeyboardInterrupt:
        logger.info("\nStopping test...")
    finally:
        # Stop training scheduler
        logger.info("Stopping training scheduler...")
        await stop_scheduler()

if __name__ == "__main__":
    asyncio.run(main()) 