# End-to-end video transcription pipeline:
# ZIP archive -> extracted audio -> five-minute chunks -> transcript files.
import json
import subprocess
import tempfile
import wave
import zipfile
from pathlib import Path, PurePosixPath

from . import config
from .backends import GoogleBackend, LocalBackend

# All backends receive the same mono, 16 kHz, 16-bit PCM audio format.
SAMPLE_RATE = 16000
# Chunking limits the amount of audio processed by one recognition request.
CHUNK_SECONDS = 5 * 60
# Only these video formats are accepted from the input archive.
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def video_entries(archive):
    """Find supported videos and reject unsafe or conflicting ZIP paths."""
    entries = []
    names = set()
    for entry in archive.infolist():
        if entry.is_dir():
            continue
        path = PurePosixPath(entry.filename)
        # Skip macOS metadata entries such as __MACOSX/._video.mp4.
        if "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        if path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        # Protect output paths from ZIP path traversal and Windows-style paths.
        if (path.is_absolute() or ".." in path.parts or
                "\\" in entry.filename or ":" in path.parts[0]):
            raise ValueError("Unsafe video path in ZIP: {!r}".format(entry.filename))
        folder = path.with_suffix("")
        # Two videos with the same stem would overwrite each other's outputs.
        if folder in names:
            raise ValueError(
                "Two videos would share output folder {!r}".format(str(folder))
            )
        names.add(folder)
        entries.append((entry, path))
    if not entries:
        raise ValueError("No supported videos found in {}".format(config.TRAINING_ZIP))
    return entries


def extract_audio(video_path, audio_path):
    """Use ffmpeg to convert a video into the format expected by the backends."""
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
            "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
            "-c:a", "pcm_s16le", str(audio_path),
        ],
        check=True,
    )


def audio_chunks(audio_path, output_dir, name):
    """Yield five-minute WAV files plus their video-time offsets and durations."""
    frames_per_chunk = SAMPLE_RATE * CHUNK_SECONDS
    with wave.open(str(audio_path), "rb") as source:
        # Validate the format before splitting so recognition settings stay correct.
        if (source.getnchannels() != 1 or source.getframerate() != SAMPLE_RATE or
                source.getsampwidth() != 2 or source.getcomptype() != "NONE"):
            raise ValueError("Expected mono 16-kHz 16-bit PCM audio")
        starts = range(0, source.getnframes(), frames_per_chunk)
        for number, start_frame in enumerate(starts, 1):
            frame_count = min(frames_per_chunk, source.getnframes() - start_frame)
            chunk_path = output_dir / "{}_part{:03d}.wav".format(name, number)
            with wave.open(str(chunk_path), "wb") as chunk:
                chunk.setparams(source.getparams())
                remaining = frame_count
                while remaining:
                    count = min(remaining, SAMPLE_RATE * 10)
                    chunk.writeframes(source.readframes(count))
                    remaining -= count
            yield chunk_path, start_frame / SAMPLE_RATE, frame_count / SAMPLE_RATE


def make_backend():
    """Create the configured backend behind a common transcribe() interface."""
    if config.BACKEND == "local":
        return LocalBackend(config.WHISPER_MODEL, config.LANGUAGE_CODE)
    if config.BACKEND == "google":
        if not config.GCS_BUCKET:
            raise ValueError("Set GCS_BUCKET in transcription_poc/config.py for Google mode")
        return GoogleBackend(config.GCS_BUCKET, config.LANGUAGE_CODE)
    raise ValueError("Unknown BACKEND: {!r}".format(config.BACKEND))


def write_transcript(base_path, video, segments, **extra):
    """Write both human-readable TXT and timestamped JSON transcript files."""
    text = "\n".join(segment["text"] for segment in segments if segment["text"])
    Path(str(base_path) + ".txt").write_text(
        text + "\n" if text else "", encoding="utf-8",
    )
    data = {"video": video}
    data.update(extra)
    data["segments"] = segments
    Path(str(base_path) + ".json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8",
    )


def transcribe_video(archive, entry, relative_path, backend):
    """Process one archived video and return its output directory."""
    name = relative_path.stem
    output_dir = config.OUTPUT_DIR / relative_path.with_suffix("")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / (name + ".wav")

    # Keep the extracted video temporary; the generated WAV is the useful output.
    with tempfile.TemporaryDirectory() as temp:
        video_path = Path(temp) / ("video" + relative_path.suffix)
        with archive.open(entry) as source, video_path.open("wb") as destination:
            while True:
                block = source.read(1024 * 1024)
                if not block:
                    break
                destination.write(block)
        extract_audio(video_path, audio_path)

    segments = []
    chunk_count = 0
    for chunk_path, offset, duration in audio_chunks(audio_path, output_dir, name):
        # Convert chunk-relative timestamps into timestamps relative to the video.
        chunk_segments = [
            {
                "start_seconds": round(offset + segment["start_seconds"], 3),
                "end_seconds": round(offset + segment["end_seconds"], 3),
                "text": segment["text"].strip(),
            }
            for segment in backend.transcribe(chunk_path)
        ]
        write_transcript(
            chunk_path.with_suffix(""), str(relative_path), chunk_segments,
            chunk=chunk_path.name, chunk_start_seconds=offset,
            chunk_end_seconds=offset + duration,
        )
        segments.extend(chunk_segments)
        chunk_count += 1
        print("  {} transcribed".format(chunk_path.name))
    if not chunk_count:
        raise ValueError("Video has no audio: {}".format(relative_path))

    # Write one combined transcript in addition to each chunk transcript.
    write_transcript(output_dir / name, str(relative_path), segments)
    return output_dir


def run():
    """Run the pipeline for every supported video in the configured ZIP archive."""
    if not config.TRAINING_ZIP.is_file():
        raise FileNotFoundError("Place the archive at {}".format(config.TRAINING_ZIP))
    with zipfile.ZipFile(config.TRAINING_ZIP) as archive:
        entries = video_entries(archive)
        backend = make_backend()
        for entry, relative_path in entries:
            print("Processing {}".format(relative_path))
            output = transcribe_video(archive, entry, relative_path, backend)
            print("Done -> {}".format(output))
