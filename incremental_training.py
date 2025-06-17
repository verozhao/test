"""Incremental training manager for Document AI."""

import asyncio
import uuid
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
from collections import defaultdict

from .client import EnhancedDocumentAIClient
from .models import (
    DocumentAIStatus,
    DocumentType,
    ProcessedDocument,
    IncrementalTrainingBatch,
    AutomatedTrainingConfig,
)

import logging

logger = logging.getLogger(__name__)


class IncrementalTrainingManager:
    """Manage incremental training for Document AI models."""

    def __init__(self, client: Optional[EnhancedDocumentAIClient] = None):
        """Initialize the incremental training manager.

        Args:
            client: Enhanced Document AI client instance.
        """
        self.client = client or EnhancedDocumentAIClient()

    async def run_incremental_training(self, processor_id: str) -> Optional[IncrementalTrainingBatch]:
        """Run incremental training with new documents.

        Args:
            processor_id: Document AI processor ID.

        Returns:
            IncrementalTrainingBatch if training was started, None otherwise.
        """
        try:
            logger.info(f"Starting incremental training check for processor {processor_id}")
            
            # Get training configuration
            config = await self._get_or_create_training_config(processor_id)
            if not config.enabled:
                logger.info("Incremental training is disabled")
                return None
            
            # Check if there's already an active training
            active_training = await IncrementalTrainingBatch.find_one({
                "processor_id": processor_id,
                "status": {"$in": [DocumentAIStatus.PENDING, DocumentAIStatus.TRAINING, DocumentAIStatus.DEPLOYING]}
            })
            
            if active_training:
                logger.info(f"Active training already exists: {active_training.batch_id}")
                return None
            
            # Get new documents for training
            training_docs = await self._select_training_documents(processor_id, config)
            
            if not training_docs:
                logger.info("Not enough new documents for training")
                return None
            
            # Create training batch
            batch = await self._create_training_batch(processor_id, training_docs)
            
            # Start training process
            asyncio.create_task(self._execute_training_pipeline(batch, config))
            
            return batch
            
        except Exception as e:
            logger.error(f"Error in incremental training: {str(e)}")
            return None

    async def _get_or_create_training_config(self, processor_id: str) -> AutomatedTrainingConfig:
        """Get or create automated training configuration.

        Args:
            processor_id: Document AI processor ID.

        Returns:
            AutomatedTrainingConfig instance.
        """
        config = await AutomatedTrainingConfig.find_one({"processor_id": processor_id})
        
        if not config:
            # Create default configuration
            config = AutomatedTrainingConfig(
                processor_id=processor_id,
                enabled=True,
                check_interval_minutes=60,
                min_documents_for_training=50,
                min_accuracy_for_deployment=0.85,
                document_types=list(DocumentType),  # All types
            )
            await config.save()
        
        return config

    async def _select_training_documents(
        self, processor_id: str, config: AutomatedTrainingConfig
    ) -> List[ProcessedDocument]:
        """Select documents for training.

        Args:
            processor_id: Document AI processor ID.
            config: Training configuration.

        Returns:
            List of documents to use for training.
        """
        # Check if this is initial training (no completed documents yet)
        completed_count = await ProcessedDocument.find({
            "processor_id": processor_id,
            "status": DocumentAIStatus.COMPLETED,
        }).count()
        
        if completed_count == 0:
            # Initial training - use PENDING documents
            logger.info("Selecting documents for initial training")
            unused_docs = await ProcessedDocument.find({
                "processor_id": processor_id,
                "status": DocumentAIStatus.PENDING,
                "document_type": {"$in": [dt.value for dt in config.document_types]},
            }).to_list()
        else:
            # Incremental training - use COMPLETED documents not yet used for training
            logger.info("Selecting documents for incremental training")
            unused_docs = await ProcessedDocument.find({
                "processor_id": processor_id,
                "status": DocumentAIStatus.COMPLETED,
                "used_for_training": False,
                "document_type": {"$in": [dt.value for dt in config.document_types]},
            }).to_list()
        
        if len(unused_docs) < config.min_documents_for_training:
            logger.info(f"Not enough documents: {len(unused_docs)} < {config.min_documents_for_training}")
            return []
        
        # Group by document type
        docs_by_type = defaultdict(list)
        for doc in unused_docs:
            docs_by_type[doc.document_type].append(doc)
        
        # For initial training, ensure we have at least 2 types with 5+ documents each
        if completed_count == 0:
            valid_types = sum(1 for docs in docs_by_type.values() if len(docs) >= 2)
            if valid_types < 1:
                logger.info(f"Initial training needs at least 1 document type with 2+ docs. Found {valid_types}")
                return []
        
        # Select balanced set of documents
        selected_docs = []
        max_per_type = min(len(unused_docs) // max(len(docs_by_type), 1), 50)
        
        for doc_type, docs in docs_by_type.items():
            # For initial training, only include types with 5+ documents
            if completed_count == 0 and len(docs) < 5:
                continue
            selected_docs.extend(docs[:max_per_type])
        
        # If we haven't reached the limit, add more documents
        if len(selected_docs) < 1000:  # Max training documents
            remaining = 1000 - len(selected_docs)
            for doc_type, docs in docs_by_type.items():
                if len(docs) > max_per_type:
                    selected_docs.extend(docs[max_per_type:max_per_type + remaining])
                    remaining -= len(docs[max_per_type:max_per_type + remaining])
                    if remaining <= 0:
                        break
        
        logger.info(f"Selected {len(selected_docs)} documents for training")
        return selected_docs[:1000]  # Ensure we don't exceed max

    async def _create_training_batch(
        self, processor_id: str, training_docs: List[ProcessedDocument]
    ) -> IncrementalTrainingBatch:
        """Create a training batch record.

        Args:
            processor_id: Document AI processor ID.
            training_docs: Documents to use for training.

        Returns:
            Created IncrementalTrainingBatch.
        """
        # Count documents by type
        type_counts = defaultdict(int)
        for doc in training_docs:
            type_counts[doc.document_type] += 1
        
        # Create batch
        batch = IncrementalTrainingBatch(
            batch_id=f"batch_{uuid.uuid4().hex[:12]}",
            processor_id=processor_id,
            model_id=f"model_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            document_ids=[doc.document_id for doc in training_docs],
            document_type_counts=dict(type_counts),
            status=DocumentAIStatus.PENDING,
        )
        await batch.save()
        
        # Mark documents as used for training
        for doc in training_docs:
            doc.used_for_training = True
            doc.training_batch_id = batch.batch_id
            doc.updated_at = datetime.now(timezone.utc)
            await doc.save()
        
        logger.info(f"Created training batch {batch.batch_id} with {len(training_docs)} documents")
        return batch

    async def _execute_training_pipeline(
        self, batch: IncrementalTrainingBatch, config: AutomatedTrainingConfig
    ):
        """Execute the full training pipeline.

        Args:
            batch: Training batch.
            config: Training configuration.
        """
        try:
            logger.info(f"Executing training pipeline for batch {batch.batch_id}")
            
            # Update status to training
            batch.status = DocumentAIStatus.TRAINING
            await batch.save()
            
            # Get document paths
            docs = await ProcessedDocument.find({
                "document_id": {"$in": batch.document_ids}
            }).to_list()
            
            training_paths = [doc.document_path for doc in docs]
            
            # Start training operation
            operation_name = await self.client.start_training_operation(
                training_documents=training_paths,
                model_display_name=f"Incremental Training {batch.model_id}",
            )
            
            batch.training_id = operation_name
            await batch.save()
            
            # Monitor training progress
            success = await self._monitor_training_operation(operation_name, batch)
            
            if not success:
                batch.status = DocumentAIStatus.TRAINING_FAILED
                await batch.save()
                return
            
            # Evaluate the model
            accuracy = await self._evaluate_model(batch)
            batch.accuracy_score = accuracy
            await batch.save()
            
            # Deploy if accuracy meets threshold
            if accuracy >= config.min_accuracy_for_deployment:
                await self._deploy_model(batch)
            else:
                logger.warning(
                    f"Model accuracy {accuracy} below threshold {config.min_accuracy_for_deployment}"
                )
                batch.status = DocumentAIStatus.TRAINED
                await batch.save()
            
        except Exception as e:
            logger.error(f"Error in training pipeline: {str(e)}")
            batch.status = DocumentAIStatus.TRAINING_FAILED
            batch.error_message = str(e)
            await batch.save()

    async def _monitor_training_operation(
        self, operation_name: str, batch: IncrementalTrainingBatch, timeout_hours: int = 2
    ) -> bool:
        """Monitor training operation until completion.

        Args:
            operation_name: Training operation name.
            batch: Training batch.
            timeout_hours: Maximum hours to wait.

        Returns:
            True if training succeeded, False otherwise.
        """
        start_time = datetime.now(timezone.utc)
        timeout = timedelta(hours=timeout_hours)
        
        while True:
            # Check timeout
            if datetime.now(timezone.utc) - start_time > timeout:
                logger.error(f"Training operation timed out after {timeout_hours} hours")
                return False
            
            # Check operation status
            status = await self.client.check_training_operation(operation_name)
            
            if status["done"]:
                if status["error"]:
                    logger.error(f"Training failed: {status['error']}")
                    batch.error_message = status["error"]
                    return False
                else:
                    logger.info(f"Training completed successfully")
                    batch.completed_at = datetime.now(timezone.utc)
                    return True
            
            # Wait before next check
            await asyncio.sleep(60)  # Check every minute

    async def _evaluate_model(self, batch: IncrementalTrainingBatch) -> float:
        """Evaluate the trained model using Document AI's evaluation capabilities.

        Args:
            batch: Training batch.

        Returns:
            Accuracy score (0-1).
        """
        try:
            # Get the processor name from the batch
            processor_name = batch.processor_name
            
            # Get evaluation metrics from Document AI
            evaluation_metrics = await self._get_processor_evaluation(processor_name)
            
            if not evaluation_metrics:
                logger.warning("No evaluation metrics available from Document AI")
                return 0.0
                
            # Calculate overall accuracy from Document AI metrics
            # This combines precision and recall for a balanced score
            precision = evaluation_metrics.get('precision', 0.0)
            recall = evaluation_metrics.get('recall', 0.0)
            
            if precision + recall == 0:
                return 0.0
                
            accuracy = (2 * precision * recall) / (precision + recall)  # F1 score
            logger.info(f"Model evaluation complete. Accuracy: {accuracy:.2%}")
            return accuracy
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            return 0.0
            
    async def _get_processor_evaluation(self, processor_name: str) -> Dict[str, float]:
        """Get evaluation metrics from Document AI for a specific processor.
        
        Args:
            processor_name: Name of the Document AI processor.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        try:
            # Get the processor
            processor = self.client.get_processor(processor_name)
            
            # Get evaluation metrics
            evaluation = processor.get_evaluation()
            
            if not evaluation:
                return {}
                
            # Extract relevant metrics
            metrics = {
                'precision': evaluation.get('precision', 0.0),
                'recall': evaluation.get('recall', 0.0),
                'f1_score': evaluation.get('f1Score', 0.0)
            }
            
            logger.info(f"Retrieved evaluation metrics for {processor_name}: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting processor evaluation: {str(e)}")
            return {}

    async def _deploy_model(self, batch: IncrementalTrainingBatch):
        """Deploy the trained model.

        Args:
            batch: Training batch.
        """
        try:
            logger.info(f"Deploying model {batch.model_id}")
            
            # Update status
            batch.status = DocumentAIStatus.DEPLOYING
            await batch.save()
            
            # Get processor version name from training operation
            # In real implementation, extract this from training operation metadata
            processor_version_name = f"{self.client.processor_path}/processorVersions/{batch.model_id}"
            
            # Deploy the processor version
            deployment_operation = await self.client.deploy_processor_version(
                processor_version_name
            )
            
            batch.deployment_id = deployment_operation
            await batch.save()
            
            # Monitor deployment
            success = await self._monitor_deployment_operation(deployment_operation, batch)
            
            if success:
                # Set as default processor version
                await self.client.set_default_processor_version(processor_version_name)
                
                batch.status = DocumentAIStatus.DEPLOYED
                batch.deployed_at = datetime.now(timezone.utc)
                await batch.save()
                
                logger.info(f"Model {batch.model_id} deployed successfully")
            else:
                batch.status = DocumentAIStatus.DEPLOYMENT_FAILED
                await batch.save()
                
        except Exception as e:
            logger.error(f"Error deploying model: {str(e)}")
            batch.status = DocumentAIStatus.DEPLOYMENT_FAILED
            batch.error_message = str(e)
            await batch.save()

    async def _monitor_deployment_operation(
        self, operation_name: str, batch: IncrementalTrainingBatch, timeout_minutes: int = 30
    ) -> bool:
        """Monitor deployment operation.

        Args:
            operation_name: Deployment operation name.
            batch: Training batch.
            timeout_minutes: Maximum minutes to wait.

        Returns:
            True if deployment succeeded, False otherwise.
        """
        start_time = datetime.now(timezone.utc)
        timeout = timedelta(minutes=timeout_minutes)
        
        while True:
            # Check timeout
            if datetime.now(timezone.utc) - start_time > timeout:
                logger.error(f"Deployment operation timed out after {timeout_minutes} minutes")
                return False
            
            # Check operation status
            status = await self.client.check_training_operation(operation_name)
            
            if status["done"]:
                if status["error"]:
                    logger.error(f"Deployment failed: {status['error']}")
                    batch.error_message = status["error"]
                    return False
                else:
                    logger.info(f"Deployment completed successfully")
                    return True
            
            # Wait before next check
            await asyncio.sleep(30)  # Check every 30 seconds

async def get_automated_training_config(processor_id: str) -> Optional[AutomatedTrainingConfig]:
    """Get automated training configuration for a processor.

    Args:
        processor_id: Document AI processor ID.

    Returns:
        AutomatedTrainingConfig if found, None otherwise.
    """
    return await AutomatedTrainingConfig.find_one({"processor_id": processor_id})