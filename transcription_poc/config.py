# Central configuration for input files, output files, and transcription mode.
from pathlib import Path

# Build paths from the project location instead of the current terminal folder.
ROOT = Path(__file__).resolve().parent.parent
TRAINING_ZIP = ROOT / "data" / "input" / "training.zip"
OUTPUT_DIR = ROOT / "data" / "output"

# Change only this setting to switch transcription backends.
BACKEND = "local"  # "local" or "google"

WHISPER_MODEL = "small"
GCS_BUCKET = ""  # Required when BACKEND = "google", e.g. "my-transcription-bucket"
LANGUAGE_CODE = "en-US"
