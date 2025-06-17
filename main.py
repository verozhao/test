"""Example FastAPI application with Document AI integration."""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def init_database():
    """Initialize MongoDB connection and Beanie ODM."""
    # Get MongoDB URL from environment or use default
    mongodb_url = os.getenv("MONGODB_URL", "mongodb://localhost:27017/document_ai")
    
    # Create Motor client
    client = AsyncIOMotorClient(mongodb_url)
    
    # Initialize Beanie with document models
    await init_beanie(
        database=client.document_ai,
        document_models=[
            ProcessedDocument,
            IncrementalTrainingBatch,
            AutomatedTrainingConfig,
        ]
    )
    
    logger.info("Database initialized successfully")