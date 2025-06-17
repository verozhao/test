# Document AI Integration

A Python package for processing documents using Google Cloud Document AI, with support for incremental training and automated document classification.

## Features

- Document processing using Google Cloud Document AI
- Support for both local and GCS-based document processing
- Automated document classification
- Incremental training support
- Flexible document type handling
- Comprehensive error handling and logging

## Prerequisites

- Python 3.8+
- Google Cloud Project with Document AI API enabled
- Service account with appropriate permissions
- Google Cloud Storage bucket (for GCS mode)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
export GCP_PROJECT_ID="tetrix-462721"
export DOCUMENT_AI_PROCESSOR_ID="2f18987b02fd93d8"
export GOOGLE_APPLICATION_CREDENTIALS="/Users/test/Downloads/tetrix-462721-71bf62848ec2.json"
export PDF_DIRECTORY=test_documents
export GCS_BUCKET_NAME="document-ai-test-veronica"
```

## Configuration

### Service Account Permissions

Your service account needs these roles:
- Document AI API User (`roles/documentai.apiUser`)
- Storage Object Admin (`roles/storage.objectAdmin`) - for GCS mode
- Storage Object Viewer (`roles/storage.objectViewer`) - for GCS mode

### Document AI Processor

1. Create a Document AI processor in your Google Cloud Console
2. Note the processor ID and project ID
3. Make sure the processor is in the same region as your GCS bucket (if using GCS mode)

### Testing

1. Place test documents in the `test_documents` directory
2. Run the test script:
```bash
python -m document_ai.test_local
```

The script will:
- Upload documents to GCS (if not in local mode)
- Process them with Document AI
- Display processing results
- Trigger incremental training if enabled and conditions are met

## Project Structure

```
document_ai/
├── __init__.py
├── api.py              # FastAPI router and endpoints
├── client.py           # Document AI client implementation
├── models.py           # Data models and schemas
├── incremental_training.py  # Incremental training logic
└── test_local.py       # Test script
```

## Modes of Operation

### Local Mode
- Processes documents directly from local files
- No GCS bucket required
- Faster for testing and development

### GCS Mode
- Uploads documents to GCS before processing
- Required for production use
- Enables better document management and tracking

### Database Mode
- Tracks processed documents in MongoDB
- Enables incremental training
- Requires MongoDB setup

## Error Handling

The package includes comprehensive error handling for:
- Document AI API errors
- GCS upload/download errors
- Invalid document types
- Processing failures
- Database errors

## Logging

Logging is configured to show:
- Processing status
- Document types and confidence scores
- Error messages and stack traces
- Training triggers and results

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.