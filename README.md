# Whisper Spoken LID

## Description

Spoken language identification (LID) over fixed-length windows of an audio track, using the language detection of [OpenAI Whisper](https://github.com/openai/whisper). The app reads the audio of an `AudioDocument` or `VideoDocument` in back-to-back windows of `windowMs` through `mmif.utils.audio_document_helper`, and runs Whisper's language detection on each window.

Whisper gives one language per window. Consecutive windows with the same most probable language become one `TimeFrame`, with `start` and `end` in milliseconds and the Whisper language code as `label`. Its `scores` list holds, in time order, the probability Whisper gave that language in each of the merged windows, so a low-confidence window inside a long segment stays visible. With `mergeWindows=false`, each window becomes its own `TimeFrame`. The view's `TimeFrame` metadata lists all language codes of the model as `labelset`.

Whisper pads or trims every input to 30 seconds, so a window is at most 30 seconds long.

## User instruction

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### System requirements

The app runs on CPU or, when available, a CUDA GPU. Model weights are downloaded on first use to the Whisper cache (`$XDG_CACHE_HOME/whisper`).

### Configurable runtime parameters

For the full list of parameters, see the app metadata in the [CLAMS App Directory](https://apps.clams.ai) or [`metadata.py`](metadata.py).

## Development

Run the tests with `pytest` from the repository root. The tests replace the Whisper model with a stub, so no model download is needed.
