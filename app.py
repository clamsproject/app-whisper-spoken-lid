"""
Whisper Spoken LID: language identification over fixed-length windows of an
audio track with the language detection of OpenAI Whisper, with the windows
read through the SDK's audio document helper. Consecutive windows with the
same language can be merged into one TimeFrame.
"""
import argparse
import logging
from itertools import chain, islice

import torch
import whisper
from clams import ClamsApp, Restifier
from metadata import MAX_WINDOW_MS
from mmif import AnnotationTypes, DocumentTypes, Mmif
from mmif.utils import audio_document_helper as adh
from whisper.tokenizer import get_tokenizer

# with a labelMap, the label of a window whose labels are mostly unmapped
_NEGATIVE_LABEL = '-'


class _WhisperLid:
    """The language detection of a Whisper model, run on in-memory
    windows."""

    # Whisper codes that are not BCP 47 language tags, and their tags
    _CANONICAL_CODES = {'jw': 'jv'}

    def __init__(self, model):
        """:param model: loaded Whisper model"""
        self._model = model
        tokenizer = get_tokenizer(model.is_multilingual,
                                  num_languages=model.num_languages)
        self._labels = [self._canonical(code)
                        for code in tokenizer.all_language_codes]

    @classmethod
    def load(cls, size):
        return cls(whisper.load_model(size))

    @property
    def labels(self):
        """All labels of the model, as BCP 47 tags."""
        return list(self._labels)

    @torch.no_grad()
    def detect(self, windows, label_map=None, min_score=0.5):
        """
        :param windows: mono ``float32`` sample arrays at 16 kHz
        :param label_map: ``{IN: OUT}``; without it, the label is the most
            probable language
        :param min_score: with ``label_map``, the minimum total of the best
            ``OUT`` label for it to be kept
        :returns: ``(label, score)`` per window
        """
        mels = torch.stack([
            whisper.log_mel_spectrogram(
                whisper.pad_or_trim(torch.from_numpy(samples)),
                n_mels=self._model.dims.n_mels)
            for samples in windows]).to(self._model.device)
        _, probs = self._model.detect_language(mels)
        results = []
        for window_probs in probs:
            window_probs = {self._canonical(code): p
                            for code, p in window_probs.items()}
            if label_map:
                results.append(
                    _pooled_label(window_probs, label_map, min_score))
            else:
                best = max(window_probs, key=window_probs.get)
                results.append((best, window_probs[best]))
        return results

    def _canonical(self, code):
        return self._CANONICAL_CODES.get(code, code)


def _pooled_label(probs, label_map, min_score):
    """
    :param probs: ``{label: probability}`` over all labels of the model
    :returns: ``(label, probability)``. The probability of each ``OUT``
        label is the sum over the labels mapped to it. The most probable
        ``OUT`` label wins if its probability is at least ``min_score``;
        otherwise the label is ``_NEGATIVE_LABEL``, with the total of the
        labels that are not mapped.
    """
    pooled = {}
    for label, p in probs.items():
        target = label_map.get(label, _NEGATIVE_LABEL)
        pooled[target] = pooled.get(target, 0.0) + p
    unmapped = pooled.pop(_NEGATIVE_LABEL, 0.0)
    # empty when every label is mapped to the negative label
    best = max(pooled, key=pooled.get) if pooled else None
    if best is not None and pooled[best] >= min_score:
        return best, pooled[best]
    return _NEGATIVE_LABEL, unmapped


def _batched(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def _segments(labeled_windows, merge=True):
    """
    :param labeled_windows: ``(start_ms, end_ms, label, score)`` per window,
        in time order
    :param merge: join consecutive windows with the same label, when each
        starts where the previous one ends
    :returns: ``(start_ms, end_ms, label, scores)`` per segment, with one
        score per window in time order
    """
    segment = None
    for start, end, label, score in labeled_windows:
        if (merge and segment is not None and segment[2] == label
                and segment[1] == start):
            segment[1] = end
            segment[3].append(score)
            continue
        if segment is not None:
            yield tuple(segment)
        segment = [start, end, label, [score]]
    if segment is not None:
        yield tuple(segment)


def _parse_label_map(entries, labelset):
    """
    :param entries: ``IN:OUT`` strings; an entry without a colon keeps the
        label as it is
    :returns: ``{IN: OUT}``
    :raises ValueError: if an ``IN`` label is not a label of the model
    """
    label_map = {}
    for entry in entries:
        source, _, target = (part.strip() for part in entry.partition(':'))
        if source not in labelset:
            raise ValueError(f'labelMap: "{source}" is not a label of the '
                             f'model; see the labelset in the app metadata')
        label_map[source] = target or source
    return label_map


def _mapped_labelset(label_map):
    labels = list(dict.fromkeys(label_map.values()))
    return labels + [_NEGATIVE_LABEL] * (_NEGATIVE_LABEL not in labels)


class WhisperSpokenLid(ClamsApp):

    def __init__(self):
        super().__init__()
        self._models = {}

    def _appmetadata(self):
        pass

    def _get_model(self, size):
        if size not in self._models:
            self.logger.debug(f'Loading Whisper model: {size}')
            self._models[size] = _WhisperLid.load(size)
        return self._models[size]

    @staticmethod
    def _label_windows(model, windows, batch_size, label_map, min_score):
        """
        :param windows: ``(samples, start_ms, end_ms)`` per window
        :returns: ``(start_ms, end_ms, label, score)`` per window
        """
        for batch in _batched(windows, batch_size):
            detected = model.detect([samples for samples, _, _ in batch],
                                    label_map, min_score)
            for (_, start_ms, end_ms), (label, score) in zip(batch,
                                                             detected):
                yield start_ms, end_ms, label, score

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        window_ms = parameters['windowMs']
        if window_ms > MAX_WINDOW_MS:
            raise ValueError(f'windowMs must be at most {MAX_WINDOW_MS}, '
                             f'got {window_ms}')
        model = self._get_model(parameters['model'])
        labels = model.labels
        label_map = _parse_label_map(parameters['labelMap'], labels)
        labelset = _mapped_labelset(label_map) if label_map else labels
        docs = chain(mmif.get_documents_by_type(DocumentTypes.AudioDocument),
                     mmif.get_documents_by_type(DocumentTypes.VideoDocument))
        for doc in docs:
            view = mmif.new_view()
            self.sign_view(view, parameters)
            view.new_contain(AnnotationTypes.TimeFrame, document=doc.id,
                             timeUnit='milliseconds', labelset=labelset)
            windows = adh.iter_audio_windows(
                doc, window_ms, sample_rate=whisper.audio.SAMPLE_RATE,
                min_last_window_ms=parameters['minLastWindowMs'])
            labeled = self._label_windows(model, windows,
                                          parameters['batchSize'],
                                          label_map,
                                          parameters['minLabelScore'])
            for start_ms, end_ms, label, scores in _segments(
                    labeled, parameters['tfMerge']):
                view.new_annotation(AnnotationTypes.TimeFrame,
                                    start=start_ms, end=end_ms,
                                    label=label, scores=scores)
        return mmif


def get_app():
    return WhisperSpokenLid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000",
                        help="set port to listen")
    parser.add_argument("--production", action="store_true",
                        help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
