"""Define models for Document AI integration."""

from datetime import datetime, timezone
from enum import Enum
from typing import List, Optional, Dict, Any

from beanie import Document, PydanticObjectId, Link, Indexed
from pydantic import BaseModel, Field, conint, confloat
from pymongo import IndexModel
from motor.motor_asyncio import AsyncIOMotorClient


class DocumentType(str, Enum):
    """Enum for document types that can be processed."""
    CAPITAL_CALL = "capital_call"
    DISTRIBUTION_NOTICE = "distribution_notice"
    FINANCIAL_STATEMENT = "financial_statement"
    INVESTMENT_OVERVIEW = "investment_overview"
    INVESTOR_MEMOS = "investor_memos"
    INVESTOR_PRESENTATION = "investor_presentation"
    INVESTOR_STATEMENT = "investor_statement"
    LEGAL = "legal"
    MANAGEMENT_COMMENTARY = "management_commentary"
    PCAP_STATEMENT = "pcap_statement"
    PORTFOLIO_SUMMARY = "portfolio_summary"
    TAX = "tax"
    OTHER = "other"


class DocumentAIStatus(str, Enum):
    """Define possible statuses for Document AI operations."""

    # Document processing statuses
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    
    # Training statuses
    TRAINING = "training"
    TRAINED = "trained"
    TRAINING_FAILED = "training_failed"
    
    # Deployment statuses
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    DEPLOYMENT_FAILED = "deployment_failed"
    
    # Feedback statuses
    FEEDBACK_PENDING = "feedback_pending"
    FEEDBACK_SUBMITTED = "feedback_submitted"
    FEEDBACK_REJECTED = "feedback_rejected"


class BudgetMode(str, Enum):
    """Define budget modes for training configuration."""

    LOW = "low_budget"
    MEDIUM = "medium_budget"
    HIGH = "high_budget"


class ProcessedDocument(Document):
    """Document that has been processed by Document AI."""
    
    document_id: str = Field(..., description="Unique document identifier")
    gcp_document_id: str = Field(..., description="GCP Document AI document ID")
    document_path: str = Field(..., description="GCS path of the document")  # Changed from gcs_path
    document_type: DocumentType = Field(..., description="Type of document")
    confidence_score: float = Field(..., description="Confidence score of the classification")  # Changed from confidence
    processor_id: str = Field(..., description="Document AI processor ID")
    status: DocumentAIStatus = Field(..., description="Processing status")
    extracted_data: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Extracted data")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    used_for_training: bool = Field(default=False, description="Whether document was used for training")
    training_batch_id: Optional[str] = Field(default=None, description="Training batch ID if used")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    class Settings:
        """Define the settings for the processed document schema."""
        name = "processed_documents"
        indexes = [
            "document_id",
            "document_path",
            "document_type",
            "processor_id",
            "status",
            "created_at",
            "used_for_training",
        ]


class DocumentAITrainingConfig(BaseModel):
    """Configuration for Document AI training."""

    model_id: str
    training_documents: List[str]
    processor_id: str
    document_types: List[DocumentType] = Field(description="Document types to train on")
    batch_size: conint(ge=1) = 100
    max_retries: conint(ge=1, le=10) = 3
    timeout_seconds: conint(ge=60) = 7200  # 2 hours
    incremental_training: bool = Field(default=True, description="Enable incremental training")
    auto_deploy: bool = Field(default=True, description="Auto deploy after successful training")
    min_documents_per_type: conint(ge=5) = 10  # Minimum docs per type for training
    min_new_documents: conint(ge=10) = 50  # Minimum new docs to trigger training
    max_training_documents: conint(ge=100) = 1000  # Max docs to use in single training


class IncrementalTrainingBatch(Document):
    """Batch of documents used for incremental training."""
    
    batch_id: str = Field(..., description="Unique batch identifier")
    processor_id: str = Field(..., description="Document AI processor ID")
    model_id: str = Field(..., description="Model identifier")
    document_ids: List[str] = Field(default_factory=list, description="IDs of documents in batch")
    document_type_counts: Dict[str, int] = Field(default_factory=dict, description="Count by document type")
    status: DocumentAIStatus = Field(default=DocumentAIStatus.PENDING, description="Training status")
    training_id: Optional[str] = Field(default=None, description="Training operation ID")
    deployment_id: Optional[str] = Field(default=None, description="Deployment operation ID")
    processor_name: Optional[str] = Field(default=None, description="Full processor resource name")
    accuracy_score: Optional[float] = Field(default=None, description="Training accuracy")  # Changed from accuracy
    started_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = Field(default=None, description="When training completed")
    deployed_at: Optional[datetime] = Field(default=None, description="When model was deployed")
    error_message: Optional[str] = Field(default=None, description="Error message if failed")
    
    class Settings:
        """Define the settings for the training batch schema."""
        name = "training_batches"
        indexes = [
            "batch_id",
            "processor_id",
            "status",
            "started_at",
        ]


class DocumentUploadRequest(BaseModel):
    """Request model for document upload."""
    
    document_name: str = Field(description="Name of the document")
    document_type: Optional[DocumentType] = Field(None, description="Expected document type")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional metadata")


class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    
    document_id: str = Field(description="Unique document identifier")
    gcp_document_id: str = Field(description="GCP Document AI document ID")
    document_path: str = Field(description="GCS path to document")
    status: DocumentAIStatus = Field(description="Processing status")
    document_type: Optional[DocumentType] = Field(None, description="Classified document type")
    confidence_score: Optional[float] = Field(None, description="Classification confidence")
    message: str = Field(description="Status message")
    training_triggered: bool = Field(default=False, description="Whether training was triggered")
    deployment_status: Optional[str] = Field(None, description="Deployment status if applicable")


class TrainingStatusResponse(BaseModel):
    """Response model for training status."""
    
    processor_id: str
    active_training: Optional[Dict[str, Any]] = None
    pending_documents: int
    last_training: Optional[Dict[str, Any]] = None
    next_training_estimate: Optional[datetime] = None
    deployed_model: Optional[Dict[str, Any]] = None
    document_type_distribution: Dict[str, int]


class AutomatedTrainingConfig(Document):
    """Configuration for automated training."""
    
    processor_id: str = Field(..., description="Document AI processor ID")
    enabled: bool = Field(default=True, description="Whether automated training is enabled")
    check_interval_minutes: int = Field(default=60, description="Minutes between training checks")
    min_documents_for_training: int = Field(default=50, description="Minimum documents to trigger training")  # Changed from min_documents
    min_accuracy_for_deployment: float = Field(default=0.85, description="Minimum accuracy to deploy")
    document_types: List[DocumentType] = Field(default_factory=list, description="Document types to include in training")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Settings:
        """Define the settings for the automated training config schema."""
        name = "training_configs"
        indexes = [
            "processor_id",
            "enabled",
        ]