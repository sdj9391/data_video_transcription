# Transcription implementations. Both backends return the same segment format
# so the rest of the pipeline does not need to know which service is being used.
import uuid


def seconds(duration):
    """Convert Google's seconds/nanoseconds duration into decimal seconds."""
    return duration.seconds + duration.nanos / 1_000_000_000


class LocalBackend:
    """Run speech-to-text locally with the faster-whisper package."""

    def __init__(self, model_name, language_code):
        # Import this optional dependency only when local mode is selected.
        from faster_whisper import WhisperModel

        # device="auto" lets Whisper choose CPU/GPU; int8 reduces memory usage.
        self.model = WhisperModel(model_name, device="auto", compute_type="int8")
        # Whisper expects the base language ("en"), while config uses "en-US".
        self.language = language_code.split("-")[0]

    def transcribe(self, chunk_path):
        """Transcribe one WAV chunk and return timestamped text segments."""
        segments, _ = self.model.transcribe(
            str(chunk_path), language=self.language, vad_filter=True,
        )
        # Convert library-specific segment objects into plain dictionaries.
        return [
            {"start_seconds": segment.start, "end_seconds": segment.end,
             "text": segment.text}
            for segment in segments
        ]


class GoogleBackend:
    """Use Google Speech-to-Text with Google Cloud Storage for audio input."""

    def __init__(self, bucket_name, language_code):
        # These packages are optional and are needed only for Google mode.
        from google.cloud import speech, storage

        self.speech = speech
        self.client = speech.SpeechClient()
        self.bucket = storage.Client().bucket(bucket_name)
        self.language_code = language_code
        # Keep temporary uploads for this run under a unique cloud prefix.
        self.run_prefix = "training-transcription/{}".format(uuid.uuid4().hex)

    def transcribe(self, chunk_path):
        """Upload, transcribe, and delete one audio chunk."""
        blob = self.bucket.blob("{}/{}.wav".format(self.run_prefix, uuid.uuid4().hex))
        blob.upload_from_filename(str(chunk_path))
        try:
            # Google recognizes audio from a Cloud Storage URI.
            audio = self.speech.RecognitionAudio(uri="gs://{}/{}".format(
                self.bucket.name, blob.name,
            ))
            # This must match the WAV files produced by pipeline.py.
            recognition = self.speech.RecognitionConfig(
                encoding=self.speech.RecognitionConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=16000,
                language_code=self.language_code,
                enable_word_time_offsets=True,
            )
            response = self.client.long_running_recognize(
                config=recognition, audio=audio,
            ).result(timeout=1800)
            segments = []
            previous_end = 0.0
            for result in response.results:
                if not result.alternatives:
                    continue
                alternative = result.alternatives[0]
                words = alternative.words
                end = seconds(result.result_end_time)
                # Use the previous end when Google does not return word timing.
                start = seconds(words[0].start_time) if words else previous_end
                segments.append({
                    "start_seconds": start,
                    "end_seconds": end,
                    "text": alternative.transcript,
                })
                previous_end = end
            return segments
        finally:
            # Always remove the temporary upload, including when recognition fails.
            blob.delete()
