"""Document AI package for processing and training documents."""

# Core client
from .client import EnhancedDocumentAIClient

# Models
from .models import (
    DocumentAIStatus,
    DocumentType,
    ProcessedDocument,
    IncrementalTrainingBatch,
    AutomatedTrainingConfig,
)

# Training components
from .incremental_training import IncrementalTrainingManager

# API and Training scheduler
from .api import router as document_ai_router
from .training_scheduler import start_scheduler, stop_scheduler, get_scheduler

__all__ = [
    # Core client
    'EnhancedDocumentAIClient',
    
    # Models
    'DocumentAIStatus',
    'DocumentType',
    'ProcessedDocument',
    'IncrementalTrainingBatch',
    'AutomatedTrainingConfig',
    
    # Training
    'IncrementalTrainingManager',
    
    # API and Scheduler
    'document_ai_router',
    'start_scheduler',
    'stop_scheduler',
    'get_scheduler',
]

# Version
__version__ = "2.0.0"
