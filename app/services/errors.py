class NotFoundError(ValueError):
    """The requested chat, message or document does not exist (HTTP 404)."""


class ConflictError(ValueError):
    """A concurrent update prevented the change; retrying is safe (HTTP 409)."""
