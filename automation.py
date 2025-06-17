"""Automation module for Document AI training configuration."""

from typing import List
from .models import AutomatedTrainingConfig, DocumentType

def get_automated_training_config() -> AutomatedTrainingConfig:
    """Get the automated training configuration.
    
    Returns:
        AutomatedTrainingConfig: The training configuration.
    """
    return AutomatedTrainingConfig(
        enabled=True,
        min_documents_for_training=2,
        processor_id="ddc065df69bfa3b5",  
        document_types=[
            DocumentType.CAPITAL_CALL,
            DocumentType.DISTRIBUTION_NOTICE,
            DocumentType.FINANCIAL_STATEMENT,
            DocumentType.INVESTMENT_OVERVIEW,
            DocumentType.INVESTOR_MEMOS,
            DocumentType.INVESTOR_PRESENTATION,
            DocumentType.INVESTOR_STATEMENT,
            DocumentType.LEGAL,
            DocumentType.MANAGEMENT_COMMENTARY,
            DocumentType.PCAP_STATEMENT,
            DocumentType.PORTFOLIO_SUMMARY,
            DocumentType.TAX,
            DocumentType.OTHER
        ]
    )