"""External How2Sign preparation pipeline."""
from .artifacts import prepare, validate  # noqa: F401
from .metadata import inspect_dataset, read_annotations, safe_id  # noqa: F401

__version__ = '0.2.0'
