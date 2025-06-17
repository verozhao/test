"""Enhanced Document AI client implementation."""

import asyncio
import os
import logging
import uuid
import traceback
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timezone

from google.cloud import documentai_v1 as documentai
from google.cloud import storage
from google.api_core import retry
from google.api_core.client_options import ClientOptions

from collections import defaultdict

from .models import (
    DocumentAIStatus,
    DocumentType,
    ProcessedDocument,
    IncrementalTrainingBatch,
    DocumentUploadResponse,
)

# Set up logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)


class EnhancedDocumentAIClient:
    """Enhanced client for Document AI with classification and training support."""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: str = "us",
        processor_id: Optional[str] = None,
        gcs_bucket: Optional[str] = None,
        local_mode: bool = False,
        skip_db: bool = False,
    ):
        """Initialize the enhanced Document AI client.

        Args:
            project_id: GCP project ID.
            location: GCP location.
            processor_id: Document AI processor ID.
            gcs_bucket: GCS bucket for document storage.
            local_mode: If True, process documents locally without GCS upload.
            skip_db: If True, skip database operations (for testing).
        """
        self.project_id = project_id or os.getenv("GCP_PROJECT_ID")
        if not self.project_id:
            raise ValueError("GCP_PROJECT_ID environment variable is not set")
            
        self.location = location
        self.processor_id = processor_id or os.getenv("DOCUMENT_AI_PROCESSOR_ID")
        if not self.processor_id:
            raise ValueError("DOCUMENT_AI_PROCESSOR_ID environment variable is not set")
            
        self.gcs_bucket = gcs_bucket or os.getenv("GCS_BUCKET", f"{self.project_id}-document-ai")
        self.local_mode = local_mode
        self.skip_db = skip_db
        
        # Initialize clients
        opts = ClientOptions(api_endpoint=f"{location}-documentai.googleapis.com")
        self.client = documentai.DocumentProcessorServiceClient(client_options=opts)
        
        if not local_mode:
            self.storage_client = storage.Client(project=self.project_id)
            # Get or create bucket
            self._ensure_bucket_exists()
        
        # Base processor path - NO VERSION SPECIFIED
        self.processor_path = f"projects/{self.project_id}/locations/{self.location}/processors/{self.processor_id}"
        
        # Check if processor has any versions
        self._check_processor_versions()


    def _ensure_bucket_exists(self):
        """Ensure the GCS bucket exists."""
        try:
            self.bucket = self.storage_client.bucket(self.gcs_bucket)
            if not self.bucket.exists():
                self.bucket = self.storage_client.create_bucket(
                    self.gcs_bucket,
                    location="US"
                )
                logger.info(f"Created GCS bucket: {self.gcs_bucket}")
        except Exception as e:
            logger.error(f"Error with GCS bucket: {str(e)}")
            raise

    def _check_processor_versions(self):
        """Check available processor versions and set default version."""
        try:
            # List processor versions
            request = documentai.ListProcessorVersionsRequest(
                parent=self.processor_path
            )
            versions = self.client.list_processor_versions(request=request)
            
            # Get the latest deployed version
            self.default_version = None
            for version in versions:
                if version.state == "DEPLOYED":  # Using string literal instead of enum
                    self.default_version = version.name
                    logger.info(f"Using deployed processor version: {version.display_name}")
                    break
            
            if not self.default_version:
                logger.warning("No deployed processor version found")
                
        except Exception as e:
            logger.error(f"Error checking processor versions: {str(e)}")
            self.default_version = None

    @retry.Retry()
    async def upload_and_process_document(
        self,
        document_path: str,
        document_name: str,
        mime_type: str = "application/pdf",
        expected_type: Optional[DocumentType] = None,
    ) -> DocumentUploadResponse:
        """Upload and process a document using Document AI.
        
        If processor has no trained versions, stores document for initial training.
        Otherwise, processes document and stores for future retraining.
        
        Args:
            document_path: Path to the document (local file path or GCS URI)
            document_name: Name of the document
            mime_type: MIME type of the document
            expected_type: Expected document type (optional)
            
        Returns:
            DocumentUploadResponse with processing results
        """
        document_id = str(uuid.uuid4())
        gcs_uri = None
        
        try:
            # Handle GCS upload
            if document_path.startswith('gs://'):
                gcs_uri = document_path
            else:
                # Upload to GCS if not in local mode
                if not self.local_mode:
                    blob_name = f"documents/{document_id}/{document_name}"
                    blob = self.bucket.blob(blob_name)
                    blob.upload_from_filename(document_path)
                    gcs_uri = f"gs://{self.gcs_bucket}/{blob_name}"
                    logger.info(f"Uploaded document to: {gcs_uri}")
                else:
                    gcs_uri = document_path

            # Check if processor has trained versions
            if self.default_version is None:
                logger.info("No trained version available - storing document for initial training")
                
                # Store document for training without processing
                if not self.skip_db:
                    processed_doc = ProcessedDocument(
                        document_id=document_id,
                        gcp_document_id=document_id,
                        document_path=gcs_uri,
                        document_type=expected_type or DocumentType.OTHER,
                        confidence_score=0.0,
                        processor_id=self.processor_id,
                        status=DocumentAIStatus.PENDING,
                        extracted_data={"note": "Stored for initial training"},
                    )
                    await processed_doc.save()
                    
                    # Check if we have enough for initial training
                    pending_count = await ProcessedDocument.find({
                        "processor_id": self.processor_id,
                        "status": DocumentAIStatus.PENDING,
                    }).count()
                    
                    logger.info(f"Documents pending for initial training: {pending_count}")
                
                return DocumentUploadResponse(
                    document_id=document_id,
                    gcp_document_id=document_id,
                    document_path=gcs_uri,
                    status=DocumentAIStatus.PENDING,
                    document_type=expected_type or DocumentType.OTHER,
                    confidence_score=0.0,
                    message=f"Document stored for initial training",
                    training_triggered=False,
                )

            # Process with Document AI using the default version
            # Read document content
            if document_path.startswith('gs://'):
                storage_client = storage.Client()
                bucket_name = gcs_uri.split('/')[2]
                blob_name = '/'.join(gcs_uri.split('/')[3:])
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                content = blob.download_as_bytes()
            else:
                with open(document_path, "rb") as f:
                    content = f.read()

            # Create process request using the versioned processor
            request = documentai.ProcessRequest(
                name=self.default_version,  # Use the specific version
                raw_document=documentai.RawDocument(
                    content=content,
                    mime_type=mime_type,
                ),
                skip_human_review=True,
            )

            # Process the document
            result = await asyncio.to_thread(
                self.client.process_document,
                request=request,
            )
            
            # Extract results
            document_type, confidence = await self._classify_document(result.document)
            extracted_data = await self._extract_document_data(result.document, document_type)

            if self.skip_db:
                return DocumentUploadResponse(
                    document_id=document_id,
                    gcp_document_id=document_id,
                    document_path=gcs_uri,
                    status=DocumentAIStatus.COMPLETED,
                    document_type=document_type,
                    confidence_score=confidence,
                    message=f"Document processed successfully",
                    training_triggered=False,
                )

            # Save processed document
            processed_doc = ProcessedDocument(
                document_id=document_id,
                gcp_document_id=document_id,
                document_path=gcs_uri,
                document_type=document_type,
                confidence_score=confidence,
                processor_id=self.processor_id,
                status=DocumentAIStatus.COMPLETED,
                extracted_data=extracted_data,
            )
            await processed_doc.save()

            # Check if we should trigger retraining
            training_triggered = await self._check_training_trigger()

            return DocumentUploadResponse(
                document_id=document_id,
                gcp_document_id=document_id,
                document_path=gcs_uri,
                status=DocumentAIStatus.COMPLETED,
                document_type=document_type,
                confidence_score=confidence,
                message=f"Document processed and stored for retraining",
                training_triggered=training_triggered,
            )
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}\n{traceback.format_exc()}")
            
            if not self.skip_db and 'document_id' in locals():
                processed_doc = ProcessedDocument(
                    document_id=document_id,
                    gcp_document_id="",
                    document_path=gcs_uri if 'gcs_uri' in locals() else "",
                    document_type=expected_type or DocumentType.OTHER,
                    confidence_score=0.0,
                    processor_id=self.processor_id,
                    status=DocumentAIStatus.FAILED,
                    error_message=str(e),
                )
                await processed_doc.save()
            
            return DocumentUploadResponse(
                document_id=document_id if 'document_id' in locals() else "",
                gcp_document_id="",
                document_path=gcs_uri if 'gcs_uri' in locals() else "",
                status=DocumentAIStatus.FAILED,
                message=f"Error: {str(e)}",
                training_triggered=False,
            )

    async def _classify_document(self, document: Any) -> Tuple[DocumentType, float]:
        """Classify a document using Document AI results.
        
        Args:
            document: Document AI document.
            
        Returns:
            Tuple of (document_type, confidence).
        """
        try:
            # Check if document has entities (classification results)
            if hasattr(document, 'entities') and document.entities:
                for entity in document.entities:
                    # Look for document type classification
                    if entity.type_:
                        entity_type = entity.type_.lower().replace(" ", "_")
                        # Try to match to our DocumentType enum
                        for doc_type in DocumentType:
                            if entity_type == doc_type.value or entity_type in doc_type.value:
                                return doc_type, entity.confidence
                        # If no exact match, but we have a classification, return OTHER with the confidence
                        if entity.confidence > 0.5:
                            logger.info(f"Unknown classification type: {entity.type_}")
                            return DocumentType.OTHER, entity.confidence
            
            # For processors that don't provide classification (or untrained classifiers)
            # Try to infer from document text or structure
            if hasattr(document, 'text') and document.text:
                text_lower = document.text.lower()
                
                # Simple keyword-based classification as fallback
                if any(keyword in text_lower for keyword in ['capital call', 'drawdown', 'commitment']):
                    return DocumentType.CAPITAL_CALL, 0.3
                elif any(keyword in text_lower for keyword in ['distribution', 'proceeds', 'realized']):
                    return DocumentType.DISTRIBUTION_NOTICE, 0.3
                elif any(keyword in text_lower for keyword in ['financial statement', 'balance sheet', 'income statement']):
                    return DocumentType.FINANCIAL_STATEMENT, 0.3
                elif any(keyword in text_lower for keyword in ['portfolio', 'holdings', 'investments']):
                    return DocumentType.PORTFOLIO_SUMMARY, 0.3
                elif any(keyword in text_lower for keyword in ['tax', 'k-1', 'schedule']):
                    return DocumentType.TAX, 0.3
            
            # Default fallback
            return DocumentType.OTHER, 0.0
            
        except Exception as e:
            logger.error(f"Error in document classification: {str(e)}")
            return DocumentType.OTHER, 0.0

    async def _extract_document_data(
        self, document: documentai.Document, document_type: DocumentType
    ) -> Dict[str, Any]:
        """Extract relevant data based on document type.

        Args:
            document: Document AI document object.
            document_type: Classified document type.

        Returns:
            Dictionary of extracted data.
        """
        extracted_data = {
            "text_length": len(document.text) if document.text else 0,
            "page_count": len(document.pages) if document.pages else 0,
            "entities": [],
            "tables": [],
        }
        
        # Extract entities
        if document.entities:
            for entity in document.entities:
                extracted_data["entities"].append({
                    "type": entity.type_,
                    "text": entity.mention_text,
                    "confidence": entity.confidence,
                })
        
        # Extract tables if present
        if document.pages:
            for page in document.pages:
                if page.tables:
                    for table in page.tables:
                        table_data = []
                        for row in table.body_rows:
                            row_data = []
                            for cell in row.cells:
                                row_data.append(cell.layout.text_segments[0].text if cell.layout.text_segments else "")
                            table_data.append(row_data)
                        extracted_data["tables"].append(table_data)
        
        # Add document-type specific extraction
        if document_type == DocumentType.financial_statement:
            # Extract financial metrics
            extracted_data["financial_metrics"] = self._extract_financial_metrics(document)
        elif document_type == DocumentType.capital_call:
            # Extract capital call details
            extracted_data["capital_call_details"] = self._extract_capital_call_details(document)
        
        return extracted_data

    def _extract_financial_metrics(self, document: documentai.Document) -> Dict[str, Any]:
        """Extract financial metrics from financial statements."""
        # Placeholder - implement actual extraction logic
        return {
            "revenue": None,
            "expenses": None,
            "net_income": None,
            "assets": None,
            "liabilities": None,
        }

    def _extract_capital_call_details(self, document: documentai.Document) -> Dict[str, Any]:
        """Extract capital call details."""
        # Placeholder - implement actual extraction logic
        return {
            "call_amount": None,
            "due_date": None,
            "fund_name": None,
            "purpose": None,
        }

    async def _check_training_trigger(self) -> bool:
        """Check if we should trigger incremental training.

        Returns:
            True if training should be triggered.
        """
        # Count new documents not used for training
        new_docs_count = await ProcessedDocument.find({
            "processor_id": self.processor_id,
            "status": DocumentAIStatus.COMPLETED,
            "used_for_training": False,
        }).count()
        
        # Get training config
        from .automation import get_automated_training_config
        config = await get_automated_training_config(self.processor_id)
        
        if config and config.enabled and new_docs_count >= config.min_documents_for_training:
            # Trigger training asynchronously
            asyncio.create_task(self._trigger_incremental_training())
            return True
        
        return False

    async def _trigger_incremental_training(self):
        """Trigger incremental training with new documents."""
        # from .training import IncrementalTrainingManager
        from .incremental_training import IncrementalTrainingManager
        manager = IncrementalTrainingManager(self)
        await manager.run_incremental_training(self.processor_id)

    @retry.Retry()
    async def start_training_operation(
        self, training_documents: List[str], model_display_name: str
    ) -> str:
        """Start a training operation in Document AI.

        Args:
            training_documents: List of GCS URIs for training documents.
            model_display_name: Display name for the trained model.

        Returns:
            Training operation ID.
        """
        try:
            # For custom classification processors, we need to prepare the training data
            # with proper labels (document types)
            
            # Get document types from the training documents
            doc_type_mapping = {}
            if not self.skip_db:
                # Get document info from database
                for gcs_uri in training_documents:
                    doc = await ProcessedDocument.find_one({"document_path": gcs_uri})
                    if doc:
                        doc_type_mapping[gcs_uri] = doc.document_type
            
            # Check processor type
            processor = self.client.get_processor(request={"name": self.processor_path})
            
            if processor.type_ == "CUSTOM_CLASSIFICATION_PROCESSOR":
                # For custom classifiers, create labeled examples
                logger.info(f"Preparing training data for custom classifier with {len(training_documents)} documents")
                
                # Create document schema with entity types for each document type
                document_schema = documentai.DocumentSchema()
                
                # Get unique document types
                unique_types = set(doc_type_mapping.values()) if doc_type_mapping else {DocumentType.OTHER}
                
                for doc_type in unique_types:
                    entity_type = documentai.DocumentSchema.EntityType(
                        type_=doc_type.value if hasattr(doc_type, 'value') else str(doc_type),
                        display_name=str(doc_type).replace("_", " ").title(),
                    )
                    document_schema.entity_types.append(entity_type)
                
                # Create labeled documents
                labeled_documents = []
                for gcs_uri in training_documents:
                    doc_type = doc_type_mapping.get(gcs_uri, DocumentType.OTHER)
                    labeled_doc = documentai.Document(
                        uri=gcs_uri,
                        type_=doc_type.value if hasattr(doc_type, 'value') else str(doc_type),
                    )
                    labeled_documents.append(labeled_doc)
                
                # Create processor version
                processor_version = documentai.ProcessorVersion(
                    display_name=model_display_name,
                    document_schema=document_schema,
                )
                
                # Prepare training request
                request = documentai.TrainProcessorVersionRequest(
                    parent=self.processor_path,
                    processor_version=processor_version,
                    document_schema=document_schema,
                    input_data=documentai.TrainProcessorVersionRequest.InputData(
                        training_documents=documentai.BatchDocumentsInputConfig(
                            gcs_documents=documentai.GcsDocuments(
                                documents=[
                                    documentai.GcsDocument(
                                        gcs_uri=uri,
                                        mime_type="application/pdf"
                                    )
                                    for uri in training_documents
                                ]
                            )
                        )
                    ),
                    base_processor_version=self.default_version if self.default_version else None,
                )
            else:
                # For other processor types, use standard training
                processor_version = documentai.ProcessorVersion(
                    display_name=model_display_name,
                )
                
                request = documentai.TrainProcessorVersionRequest(
                    parent=self.processor_path,
                    processor_version=processor_version,
                    input_data=documentai.TrainProcessorVersionRequest.InputData(
                        training_documents=documentai.BatchDocumentsInputConfig(
                            gcs_documents=documentai.GcsDocuments(
                                documents=[
                                    documentai.GcsDocument(
                                        gcs_uri=uri,
                                        mime_type="application/pdf"
                                    )
                                    for uri in training_documents
                                ]
                            )
                        )
                    ),
                    base_processor_version=self.default_version if self.default_version else None,
                )
            
            # Start training
            logger.info(f"Starting training operation for {len(training_documents)} documents")
            operation = await asyncio.to_thread(
                self.client.train_processor_version,
                request=request,
            )
            
            logger.info(f"Training operation started: {operation.name}")
            return operation.name
            
        except Exception as e:
            logger.error(f"Error starting training operation: {str(e)}")
            raise

    @retry.Retry()
    async def check_training_operation(self, operation_name: str) -> Dict[str, Any]:
        """Check the status of a training operation.

        Args:
            operation_name: Training operation name.

        Returns:
            Dictionary with operation status and details.
        """
        operation = await asyncio.to_thread(
            self.client._transport.operations_client.get_operation,
            name=operation_name,
        )
        
        return {
            "done": operation.done,
            "error": str(operation.error) if operation.error else None,
            "metadata": operation.metadata,
        }

    @retry.Retry()
    async def deploy_processor_version(self, processor_version_name: str) -> str:
        """Deploy a trained processor version.

        Args:
            processor_version_name: Name of the processor version to deploy.

        Returns:
            Deployment operation name.
        """
        request = documentai.DeployProcessorVersionRequest(
            name=processor_version_name,
        )
        
        operation = await asyncio.to_thread(
            self.client.deploy_processor_version,
            request=request,
        )
        
        return operation.name

    @retry.Retry()
    async def set_default_processor_version(self, processor_version_name: str):
        """Set a processor version as the default.

        Args:
            processor_version_name: Name of the processor version.
        """
        processor = await asyncio.to_thread(
            self.client.get_processor,
            request=documentai.GetProcessorRequest(name=self.processor_path),
        )
        
        processor.default_processor_version = processor_version_name
        
        await asyncio.to_thread(
            self.client.update_processor,
            request=documentai.UpdateProcessorRequest(
                processor=processor,
                update_mask={"paths": ["default_processor_version"]},
            ),
        )

    async def process_document(self, gcs_path: str) -> Any:
        """Process a document using Document AI.
        
        Args:
            gcs_path: GCS path of the document.
            
        Returns:
            Document AI processing result.
        """
        try:
            # Get the processor
            processor = self.client.get_processor(self.processor_id)
            
            # Get the latest processor version
            processor_version = processor.get_latest_version()
            
            # Process the document
            result = await asyncio.to_thread(
                processor_version.process_document,
                gcs_path=gcs_path
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise
