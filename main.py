# Application entry point. The workflow itself lives in transcription_poc.pipeline.
from transcription_poc.pipeline import run


if __name__ == "__main__":
    # This guard prevents execution when main.py is imported by another module.
    run()
