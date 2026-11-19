"""
Domain Exceptions for the ANPR System
=====================================
Structured exception hierarchy for fault tolerance, logging, and API error mappings.
"""


class ANPRError(Exception):
    """Base exception for all ANPR system errors."""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class CameraError(ANPRError):
    """Raised on camera initialization, disconnection, or frame read failure."""
    pass


class CameraDisconnectedError(CameraError):
    """Raised when camera stream suddenly terminates or is unplugged."""
    pass


class ModelError(ANPRError):
    """Raised when YOLO weights are missing, corrupt, or inference fails."""
    pass


class WeightsNotFoundError(ModelError):
    """Raised when specified weight file does not exist on disk."""
    pass


class OCRError(ANPRError):
    """Raised when EasyOCR reader fails initialization or character recognition."""
    pass


class InvalidCropError(ANPRError):
    """Raised when license plate bounding box crop is empty or out-of-bounds."""
    pass


class ValidationError(ANPRError):
    """Raised on invalid registration number format or data validation issue."""
    pass


class DatabaseError(ANPRError):
    """Raised on database connection failure, lock contention, or transaction rollback."""
    pass


class DatabaseLockedError(DatabaseError):
    """Raised specifically when SQLite database reports database is locked."""
    pass


class StorageError(ANPRError):
    """Raised when disk storage is exhausted or image file cannot be saved."""
    pass


class NotificationError(ANPRError):
    """Raised when ntfy.sh dispatch fails after retries."""
    pass


class StateTransitionError(ANPRError):
    """Raised when an illegal or conflicting vehicle state transition is requested."""
    pass
