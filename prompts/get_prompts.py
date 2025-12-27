"""
Load prompt templates from YAML file for BOQ extraction.
"""
from pathlib import Path
import yaml

# Load templates from YAML
_TEMPLATES_PATH = Path(__file__).parent / "templates.yaml"

with open(_TEMPLATES_PATH, 'r', encoding='utf-8') as f:
    _templates = yaml.safe_load(f)

# Export as module constants
QA_TEMPLATE = _templates['qa_template']
METADATA_EXTRACTION_TEMPLATE = _templates['metadata_extraction_template']
BOQ_EXTRACTION_TEMPLATE = _templates['boq_extraction_template']
BOQ_COLUMN_HEADERS = _templates['boq_column_headers']