"""
Whisper Spoken LID: language identification over fixed-length windows of an
audio track, with the windows read through the SDK's audio document helper.
Consecutive windows with the same most probable language become one
TimeFrame.
"""
import argparse
import logging
from itertools import chain, groupby, islice

import torch
import whisper
from clams import ClamsApp, Restifier
from metadata import MAX_WINDOW_MS
from mmif import AnnotationTypes, DocumentTypes, Mmif
from mmif.utils import audio_document_helper as adh
from whisper.tokenizer import get_tokenizer


def batched(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


def segments(labeled_windows, merge=True):
    """
    :param labeled_windows: ``(start_ms, end_ms, label, score)`` per window,
        in time order, each window starting where the previous one ends
    :param merge: join consecutive windows with the same label
    :returns: ``(start_ms, end_ms, label, scores)`` per segment, with one
        score per window in time order
    """
    if merge:
        groups = (list(group) for _, group
                  in groupby(labeled_windows, key=lambda w: w[2]))
    else:
        groups = ([window] for window in labeled_windows)
    for group in groups:
        yield (group[0][0], group[-1][1], group[0][2],
               [window[3] for window in group])


class WhisperSpokenLid(ClamsApp):

    def __init__(self):
        super().__init__()
        self._models = {}

    def _appmetadata(self):
        pass

    def _get_model(self, size):
        if size not in self._models:
            self.logger.debug(f'Loading Whisper model: {size}')
            self._models[size] = whisper.load_model(size)
        return self._models[size]

    @staticmethod
    def detect(model, windows):
        """
        :param windows: mono 16 kHz sample arrays
        :returns: one ``{language code: probability}`` dict per window
        """
        mels = torch.stack([
            whisper.log_mel_spectrogram(
                whisper.pad_or_trim(torch.from_numpy(samples)),
                n_mels=model.dims.n_mels)
            for samples in windows]).to(model.device)
        with torch.no_grad():
            _, probs = model.detect_language(mels)
        return probs

    def label_windows(self, model, windows, batch_size):
        """
        :param windows: ``(samples, start_ms, end_ms)`` per window
        :returns: ``(start_ms, end_ms, label, score)`` per window, where
            ``label`` is the most probable language and ``score`` its
            probability
        """
        for batch in batched(windows, batch_size):
            probs = self.detect(model, [samples for samples, _, _ in batch])
            for (_, start_ms, end_ms), window_probs in zip(batch, probs):
                label = max(window_probs, key=window_probs.get)
                # TODO: only the top language's probability is kept, so a
                # close runner-up (e.g. tl 0.47 vs id 0.33) is lost; revisit
                # if consumers need the margin to the second language
                yield start_ms, end_ms, label, window_probs[label]

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        window_ms = parameters['windowMs']
        if window_ms > MAX_WINDOW_MS:
            raise ValueError(f'windowMs must be at most {MAX_WINDOW_MS}, '
                             f'got {window_ms}')
        model = self._get_model(parameters['model'])
        tokenizer = get_tokenizer(model.is_multilingual,
                                  num_languages=model.num_languages)
        labelset = list(tokenizer.all_language_codes)

        docs = chain(mmif.get_documents_by_type(DocumentTypes.AudioDocument),
                     mmif.get_documents_by_type(DocumentTypes.VideoDocument))
        for doc in docs:
            view = mmif.new_view()
            self.sign_view(view, parameters)
            view.new_contain(AnnotationTypes.TimeFrame, document=doc.id,
                             timeUnit='milliseconds', labelset=labelset)
            # windows follow each other with no overlap and no gap, so
            # merged windows cover their time span exactly
            windows = adh.iter_audio_windows(
                doc, window_ms, sample_rate=whisper.audio.SAMPLE_RATE,
                min_last_window_ms=parameters['minLastWindowMs'])
            labeled = self.label_windows(model, windows,
                                         parameters['batchSize'])
            for start_ms, end_ms, label, scores in segments(
                    labeled, parameters['mergeWindows']):
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
