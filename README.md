# Whisper Spoken LID

## Description

Spoken language identification (LID) over fixed-length windows of an audio track, using the language detection of [OpenAI Whisper](https://github.com/openai/whisper). The app reads the audio of an `AudioDocument` or `VideoDocument` in back-to-back windows of `windowMs` through `mmif.utils.audio_document_helper`, and runs Whisper's language detection on each window.

Whisper gives one label per window: its most probable language. The app writes languages as BCP 47 language tags: the two-letter ISO 639-1 code where one exists, else the three-letter ISO 639-3 code (`haw`, `yue`). Whisper's own `jw` becomes `jv` (Javanese). Consecutive windows with the same label become one `TimeFrame`, with `start` and `end` in milliseconds. Its `scores` list holds, in time order, the Whisper probability of that label in each of the merged windows, so a low-confidence window inside a long segment stays visible. With `tfMerge=false`, each window becomes its own `TimeFrame`. The view's `TimeFrame` metadata lists all labels of the model as `labelset`.

Whisper pads or trims every input to 30 seconds, so a window is at most 30 seconds long.

### Keeping only some labels

With `labelMap`, the app keeps only the labels you name, for example to tell English from everything else. Each entry is `IN:OUT`, which renames the label `IN` to `OUT`, or just `IN`, which keeps it as it is; several labels can map to the same `OUT`. All other labels become `-`.

The label of a window then comes from Whisper's probabilities for all of its languages in that window. The probabilities of the labels mapped to the same `OUT` are added up, and the window gets the `OUT` label with the largest total if that total is at least `minLabelScore` (default 0.5); otherwise it gets `-`. The score of a window is the total of its label; for `-`, the total of all labels that are not mapped. For example, with `labelMap=en`, a window is `en` when Whisper gives English at least 0.5, and `-` otherwise. A lower `minLabelScore` labels more windows `en`, a higher one fewer.

`OUT` labels are scored separately, so with `labelMap=en es` English and Spanish each have to reach `minLabelScore` on their own. To count several labels together, map them to the same `OUT`, as in `labelMap=en:target es:target`.

The probabilities are Whisper's softmax over its language tokens. They are not calibrated, so a total of 0.6 does not mean the label is right 60% of the time, and the best `minLabelScore` for a collection is best found on labeled data. The `labelset` lists the `OUT` labels and `-`.

## User instruction

General user instructions for CLAMS apps are available at [CLAMS Apps documentation](https://apps.clams.ai/clamsapp).

### System requirements

The app runs on CPU or, when available, a CUDA GPU. Model weights are downloaded on first use to the Whisper cache (`$XDG_CACHE_HOME/whisper`).

### Configurable runtime parameters

For the full list of parameters, see the app metadata in the [CLAMS App Directory](https://apps.clams.ai) or [`metadata.py`](metadata.py).

## Development

Run the tests with `pytest` from the repository root. The tests replace the Whisper model with a stub, so no model download is needed.
