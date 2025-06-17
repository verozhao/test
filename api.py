"""Enhanced API router for Document AI with full automation."""

import os
import tempfile
from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, Form, Query
from fastapi.responses import JSONResponse

from .client import EnhancedDocumentAIClient
from .incremental_training import IncrementalTrainingManager
from .models import (
    DocumentAIStatus,
    DocumentType,
    ProcessedDocument,
    IncrementalTrainingBatch,
    AutomatedTrainingConfig,
    DocumentUploadRequest,
    DocumentUploadResponse,
    TrainingStatusResponse,
)

import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/document-ai", tags=["document-ai"])


def get_client() -> EnhancedDocumentAIClient:
    """Get Document AI client instance."""
    return EnhancedDocumentAIClient()


def get_training_manager() -> IncrementalTrainingManager:
    """Get training manager instance."""
    return IncrementalTrainingManager()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    document_name: Optional[str] = Form(None),
    expected_type: Optional[DocumentType] = Form(None),
    client: EnhancedDocumentAIClient = Depends(get_client),
):
    """Upload a document to GCP and process it.

    This endpoint:
    1. Uploads the document to GCS
    2. Processes it with Document AI
    3. Classifies the document type
    4. Extracts relevant data
    5. Triggers incremental training if threshold is met

    Args:
        file: The document file to upload.
        document_name: Optional custom name for the document.
        expected_type: Optional expected document type.
        client: Document AI client instance.

    Returns:
        DocumentUploadResponse with processing results.
    """
    try:
        # Validate file type
        if file.content_type not in ["application/pdf", "image/png", "image/jpeg"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.content_type}. Only PDF, PNG, and JPEG are supported."
            )
        
        # Use provided name or original filename
        doc_name = document_name or file.filename
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            try:
                # Upload and process
                response = await client.upload_and_process_document(
                    document_path=tmp_file.name,
                    document_name=doc_name,
                    expected_type=expected_type,
                    mime_type=file.content_type,
                )
                
                return response
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_file.name)
                
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-batch")
async def upload_documents_batch(
    files: List[UploadFile] = File(...),
    expected_type: Optional[DocumentType] = Form(None),
    client: EnhancedDocumentAIClient = Depends(get_client),
):
    """Upload multiple documents in batch.

    Args:
        files: List of document files to upload.
        expected_type: Optional expected document type for all files.
        client: Document AI client instance.

    Returns:
        List of upload responses.
    """
    try:
        results = []
        
        for file in files:
            # Process each file
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file.flush()
                
                try:
                    response = await client.upload_and_process_document(
                        document_path=tmp_file.name,
                        document_name=file.filename,
                        expected_type=expected_type,
                        mime_type=file.content_type,
                    )
                    results.append(response.dict())
                except Exception as e:
                    results.append({
                        "filename": file.filename,
                        "status": "failed",
                        "error": str(e),
                    })
                finally:
                    os.unlink(tmp_file.name)
        
        return JSONResponse(content={
            "total": len(files),
            "successful": sum(1 for r in results if r.get("status") != "failed"),
            "failed": sum(1 for r in results if r.get("status") == "failed"),
            "results": results,
        })
        
    except Exception as e:
        logger.error(f"Error in batch upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/trigger-training/{processor_id}")
async def trigger_incremental_training(
    processor_id: str,
    manager: IncrementalTrainingManager = Depends(get_training_manager),
):
    """Manually trigger incremental training for a processor.

    Args:
        processor_id: Document AI processor ID.
        manager: Training manager instance.

    Returns:
        Training batch information.
    """
    try:
        batch = await manager.run_incremental_training(processor_id)
        
        if not batch:
            return JSONResponse(content={
                "message": "Training not triggered",
                "reason": "Not enough new documents or training already in progress",
            })
        
        return JSONResponse(content={
            "message": "Training triggered successfully",
            "batch_id": batch.batch_id,
            "model_id": batch.model_id,
            "document_count": len(batch.document_ids),
            "document_types": batch.document_type_counts,
            "status": batch.status,
        })
        
    except Exception as e:
        logger.error(f"Error triggering training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/status/{processor_id}", response_model=TrainingStatusResponse)
async def get_training_status(processor_id: str):
    """Get comprehensive training status for a processor.

    Args:
        processor_id: Document AI processor ID.

    Returns:
        Training status response.
    """
    try:
        # Get active training
        active_training = await IncrementalTrainingBatch.find_one({
            "processor_id": processor_id,
            "status": {"$in": [DocumentAIStatus.PENDING, DocumentAIStatus.TRAINING, DocumentAIStatus.DEPLOYING]}
        })
        
        # Get last completed training
        last_training = await IncrementalTrainingBatch.find_one({
            "processor_id": processor_id,
            "status": {"$in": [DocumentAIStatus.DEPLOYED, DocumentAIStatus.TRAINED]}
        }).sort([("completed_at", -1)])
        
        # Get deployed model
        deployed_model = await IncrementalTrainingBatch.find_one({
            "processor_id": processor_id,
            "status": DocumentAIStatus.DEPLOYED
        }).sort([("deployed_at", -1)])
        
        # Count pending documents
        pending_docs = await ProcessedDocument.find({
            "processor_id": processor_id,
            "status": DocumentAIStatus.COMPLETED,
            "used_for_training": False,
        }).count()
        
        # Get document type distribution
        pipeline = [
            {"$match": {
                "processor_id": processor_id,
                "status": DocumentAIStatus.COMPLETED,
            }},
            {"$group": {
                "_id": "$document_type",
                "count": {"$sum": 1}
            }}
        ]
        
        type_distribution = {}
        async for result in ProcessedDocument.aggregate(pipeline):
            type_distribution[result["_id"]] = result["count"]
        
        # Get training config
        config = await AutomatedTrainingConfig.find_one({"processor_id": processor_id})
        
        # Estimate next training
        next_training = None
        if config and config.enabled and pending_docs >= config.min_documents_for_training:
            next_training = datetime.now(timezone.utc)
        
        return TrainingStatusResponse(
            processor_id=processor_id,
            active_training={
                "batch_id": active_training.batch_id,
                "status": active_training.status,
                "started_at": active_training.started_at,
                "document_count": len(active_training.document_ids),
            } if active_training else None,
            pending_documents=pending_docs,
            last_training={
                "batch_id": last_training.batch_id,
                "completed_at": last_training.completed_at,
                "accuracy": last_training.accuracy_score,
                "document_count": len(last_training.document_ids),
            } if last_training else None,
            next_training_estimate=next_training,
            deployed_model={
                "batch_id": deployed_model.batch_id,
                "model_id": deployed_model.model_id,
                "deployed_at": deployed_model.deployed_at,
                "accuracy": deployed_model.accuracy_score,
            } if deployed_model else None,
            document_type_distribution=type_distribution,
        )
        
    except Exception as e:
        logger.error(f"Error getting training status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents")
async def list_documents(
    processor_id: str,
    document_type: Optional[DocumentType] = None,
    status: Optional[DocumentAIStatus] = None,
    used_for_training: Optional[bool] = None,
    limit: int = Query(100, ge=1, le=1000),
    skip: int = Query(0, ge=0),
):
    """List processed documents with filtering.

    Args:
        processor_id: Document AI processor ID.
        document_type: Filter by document type.
        status: Filter by processing status.
        used_for_training: Filter by training usage.
        limit: Maximum number of documents to return.
        skip: Number of documents to skip.

    Returns:
        List of documents and metadata.
    """
    try:
        # Build query
        query = {"processor_id": processor_id}
        if document_type:
            query["document_type"] = document_type
        if status:
            query["status"] = status
        if used_for_training is not None:
            query["used_for_training"] = used_for_training
        
        # Get documents
        documents = await ProcessedDocument.find(query).skip(skip).limit(limit).to_list()
        total = await ProcessedDocument.find(query).count()
        
        return JSONResponse(content={
            "total": total,
            "skip": skip,
            "limit": limit,
            "documents": [
                {
                    "document_id": doc.document_id,
                    "document_type": doc.document_type,
                    "confidence_score": doc.confidence_score,
                    "status": doc.status,
                    "used_for_training": doc.used_for_training,
                    "training_batch_id": doc.training_batch_id,
                    "created_at": doc.created_at.isoformat(),
                    "document_path": doc.document_path,
                }
                for doc in documents
            ],
        })
        
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/training/config/{processor_id}")
async def update_training_configuration(
    processor_id: str,
    enabled: Optional[bool] = None,
    check_interval_minutes: Optional[int] = None,
    min_documents_for_training: Optional[int] = None,
    min_accuracy_for_deployment: Optional[float] = None,
    document_types: Optional[List[DocumentType]] = None,
):
    """Update automated training configuration.

    Args:
        processor_id: Document AI processor ID.
        enabled: Whether automated training is enabled.
        check_interval_minutes: Minutes between training checks.
        min_documents_for_training: Minimum documents to trigger training.
        min_accuracy_for_deployment: Minimum accuracy to deploy.
        document_types: Document types to include in training.

    Returns:
        Updated configuration.
    """
    try:
        config = await AutomatedTrainingConfig.find_one({"processor_id": processor_id})
        
        if not config:
            config = AutomatedTrainingConfig(processor_id=processor_id)
        
        # Update fields
        if enabled is not None:
            config.enabled = enabled
        if check_interval_minutes is not None:
            config.check_interval_minutes = check_interval_minutes
        if min_documents_for_training is not None:
            config.min_documents_for_training = min_documents_for_training
        if min_accuracy_for_deployment is not None:
            config.min_accuracy_for_deployment = min_accuracy_for_deployment
        if document_types is not None:
            config.document_types = document_types
        
        config.updated_at = datetime.now(timezone.utc)
        await config.save()
        
        return JSONResponse(content={
            "message": "Configuration updated successfully",
            "config": {
                "processor_id": config.processor_id,
                "enabled": config.enabled,
                "check_interval_minutes": config.check_interval_minutes,
                "min_documents_for_training": config.min_documents_for_training,
                "min_accuracy_for_deployment": config.min_accuracy_for_deployment,
                "document_types": config.document_types,
            }
        })
        
    except Exception as e:
        logger.error(f"Error updating configuration: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/training/history/{processor_id}")
async def get_training_history(
    processor_id: str,
    limit: int = Query(10, ge=1, le=100),
):
    """Get training history for a processor.

    Args:
        processor_id: Document AI processor ID.
        limit: Maximum number of training batches to return.

    Returns:
        List of training batches.
    """
    try:
        batches = await IncrementalTrainingBatch.find({
            "processor_id": processor_id
        }).sort([("started_at", -1)]).limit(limit).to_list()
        
        return JSONResponse(content={
            "processor_id": processor_id,
            "training_batches": [
                {
                    "batch_id": batch.batch_id,
                    "model_id": batch.model_id,
                    "status": batch.status,
                    "document_count": len(batch.document_ids),
                    "document_types": batch.document_type_counts,
                    "accuracy": batch.accuracy_score,
                    "started_at": batch.started_at.isoformat(),
                    "completed_at": batch.completed_at.isoformat() if batch.completed_at else None,
                    "deployed_at": batch.deployed_at.isoformat() if batch.deployed_at else None,
                    "error": batch.error_message,
                }
                for batch in batches
            ],
        })
        
    except Exception as e:
        logger.error(f"Error getting training history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))