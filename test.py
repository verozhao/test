"""
Example FastAPI application with Document AI integration.

This file demonstrates how to set up a complete Document AI service with:
- Automatic document processing
- Incremental training
- Background scheduling
- REST API endpoints
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from beanie import init_beanie
from motor.motor_asyncio import AsyncIOMotorClient

# Import Document AI components
from document_ai import (
    document_ai_router,
    start_scheduler,
    stop_scheduler,
    ProcessedDocument,
    IncrementalTrainingBatch,
    AutomatedTrainingConfig,
    DocumentType,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def init_database():
    """Initialize MongoDB connection and Beanie ODM."""
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/document_ai")
    
    client = AsyncIOMotorClient(mongodb_url)
    
    await init_beanie(
        database=client.document_ai,
        document_models=[
            ProcessedDocument,
            IncrementalTrainingBatch,
            AutomatedTrainingConfig,
        ]
    )
    
    logger.info("Database initialized successfully")


async def create_default_config():
    """Create default training configuration if not exists."""
    processor_id = os.getenv("DOCUMENT_AI_PROCESSOR_ID")
    if not processor_id:
        logger.warning("DOCUMENT_AI_PROCESSOR_ID not set, skipping default config")
        return
    
    config = await AutomatedTrainingConfig.find_one({"processor_id": processor_id})
    
    if not config:
        config = AutomatedTrainingConfig(
            processor_id=processor_id,
            enabled=True,
            check_interval_minutes=60,
            min_documents_for_training=2,
            min_accuracy_for_deployment=0.1,
            document_types=list(DocumentType),
        )
        await config.save()
        logger.info(f"Created default training config for processor {processor_id}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting Document AI service...")
    
    try:
        # Initialize database
        await init_database()
        
        # Create default configuration
        await create_default_config()
        
        # Start training scheduler
        await start_scheduler()
        logger.info("Training scheduler started")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Document AI service...")
    await stop_scheduler()
    logger.info("Training scheduler stopped")


# Create FastAPI app
app = FastAPI(
    title="Document AI Service",
    description="Financial document processing with automated training",
    version="2.0.0",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Document AI router
app.include_router(document_ai_router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "Document AI Service",
        "version": "2.0.0",
        "status": "operational",
        "endpoints": {
            "upload": "/api/v1/document-ai/upload",
            "batch_upload": "/api/v1/document-ai/upload-batch",
            "training_status": "/api/v1/document-ai/training/status/{processor_id}",
            "docs": "/docs",
        }
    }

# Health check
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check database connection
        doc_count = await ProcessedDocument.count()
        
        # Check scheduler status
        from document_ai import get_scheduler
        scheduler = await get_scheduler()
        scheduler_status = scheduler.get_status()
        
        return {
            "status": "healthy",
            "database": "connected",
            "documents_processed": doc_count,
            "scheduler": scheduler_status,
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))

# Statistics endpoint
@app.get("/api/v1/statistics")
async def get_statistics():
    """Get service statistics."""
    try:
        # Document statistics
        total_docs = await ProcessedDocument.count()
        docs_by_type = {}
        
        pipeline = [
            {"$group": {
                "_id": "$document_type",
                "count": {"$sum": 1}
            }}
        ]
        
        async for result in ProcessedDocument.aggregate(pipeline):
            docs_by_type[result["_id"]] = result["count"]
        
        # Training statistics
        total_trainings = await IncrementalTrainingBatch.count()
        successful_trainings = await IncrementalTrainingBatch.find({
            "status": "deployed"
        }).count()
        
        return {
            "documents": {
                "total": total_docs,
                "by_type": docs_by_type,
            },
            "training": {
                "total_batches": total_trainings,
                "successful_deployments": successful_trainings,
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 8000))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    # Run the application
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )