"""
Custom Exceptions for SynapseAI
"""
from fastapi import HTTPException, status


class SynapseException(Exception):
    """Base exception for SynapseAI."""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class DocumentNotFoundError(SynapseException):
    """Raised when a document is not found."""
    pass


class DocumentProcessingError(SynapseException):
    """Raised when document processing fails."""
    pass


class AgentExecutionError(SynapseException):
    """Raised when an agent fails to execute."""
    pass


class LLMError(SynapseException):
    """Raised when LLM call fails."""
    pass


class InvalidModeError(SynapseException):
    """Raised when an invalid mode is specified."""
    pass


def raise_not_found(message: str = "Resource not found"):
    """Raise HTTP 404 exception."""
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=message)


def raise_bad_request(message: str = "Bad request"):
    """Raise HTTP 400 exception."""
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)


def raise_server_error(message: str = "Internal server error"):
    """Raise HTTP 500 exception."""
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=message)
