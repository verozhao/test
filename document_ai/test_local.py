"""Test script for Document AI processing with GCS and incremental training."""

import asyncio
import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from google.cloud import storage
from motor.motor_asyncio import AsyncIOMotorClient
from beanie import init_beanie
from pymongo.errors import ServerSelectionTimeoutError

from document_ai import (
    EnhancedDocumentAIClient,
    DocumentType,
    ProcessedDocument,
    AutomatedTrainingConfig,
    IncrementalTrainingBatch,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def upload_to_gcs(local_file_path: str, bucket_name: str) -> str:
    """Upload a file to Google Cloud Storage.
    
    Args:
        local_file_path: Path to the local file
        bucket_name: Name of the GCS bucket
        
    Returns:
        GCS URI of the uploaded file
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Create a blob name from the file name
    blob_name = Path(local_file_path).name
    blob = bucket.blob(blob_name)
    
    # Upload the file
    blob.upload_from_filename(local_file_path)
    
    # Return the GCS URI
    return f"gs://{bucket_name}/{blob_name}"

async def init_db():
    """Initialize MongoDB connection and models."""
    try:
        # Create MongoDB client
        client = AsyncIOMotorClient("mongodb://localhost:27017")
        
        # Initialize Beanie with the document models
        await init_beanie(
            database=client.document_ai,
            document_models=[
                ProcessedDocument,
                IncrementalTrainingBatch,
                AutomatedTrainingConfig
            ]
        )
        logger.info("Successfully initialized database connection")
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

async def check_training_status():
    """Check the current training status."""
    try:
        config = await AutomatedTrainingConfig.find_one({"processor_id": os.getenv("DOCUMENT_AI_PROCESSOR_ID")})
        if config:
            logger.info(f"Training Configuration:")
            logger.info(f"  Enabled: {config.enabled}")
            logger.info(f"  Min Documents: {config.min_documents_for_training}")
            logger.info(f"  Document Types: {config.document_types}")
        
        # Count processed documents
        total_docs = await ProcessedDocument.find().count()
        logger.info(f"Total processed documents: {total_docs}")
        
        # Count by document type
        for doc_type in DocumentType:
            count = await ProcessedDocument.find({"document_type": doc_type}).count()
            logger.info(f"  {doc_type}: {count} documents")
    except Exception as e:
        logger.warning(f"Could not check training status: {str(e)}")

async def process_documents(pdf_directory: str, bucket_name: str, use_db: bool = True):
    """Process all PDF documents in the specified directory."""
    # Initialize client with GCS support and training enabled
    client = EnhancedDocumentAIClient(
        project_id=os.getenv("GCP_PROJECT_ID"),
        processor_id=os.getenv("DOCUMENT_AI_PROCESSOR_ID"),
        local_mode=False,  # Use GCS mode
        skip_db=not use_db  # Skip database operations if not using DB
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
    
    # Check training status after processing if using DB
    if use_db:
        await check_training_status()

async def main():
    """Main function to process documents."""
    # Initialize MongoDB
    await init_db()
    
    # Get environment variables
    project_id = os.getenv("GCP_PROJECT_ID")
    processor_id = os.getenv("DOCUMENT_AI_PROCESSOR_ID")
    
    if not project_id:
        raise ValueError("GCP_PROJECT_ID environment variable is not set")
    if not processor_id:
        raise ValueError("DOCUMENT_AI_PROCESSOR_ID environment variable is not set")
        
    logger.info(f"Using project ID: {project_id}")
    logger.info(f"Using processor ID: {processor_id}")
        
    # Set up paths and bucket
    pdf_dir = Path("test_documents")
    bucket_name = os.getenv("GCS_BUCKET_NAME", "document-ai-test-veronica")
    use_db = True  # Enable MongoDB storage
    
    # Process documents
    await process_documents(pdf_dir, bucket_name, use_db)

if __name__ == "__main__":
    asyncio.run(main()) 
