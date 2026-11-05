"""
Whisper Spoken LID: language identification over fixed-length windows of an
audio track, with the windows read through the SDK's audio document helper.
"""
import argparse
import logging
from itertools import chain, islice

import torch
import whisper
from clams import ClamsApp, Restifier
from mmif import Mmif, AnnotationTypes, DocumentTypes
from mmif.utils import audio_document_helper as adh
from whisper.tokenizer import get_tokenizer

from metadata import MAX_WINDOW_MS


def batched(iterable, size):
    iterator = iter(iterable)
    while batch := list(islice(iterator, size)):
        yield batch


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
                whisper.pad_or_trim(torch.from_numpy(samples)), n_mels=model.dims.n_mels)
            for samples in windows]).to(model.device)
        with torch.no_grad():
            _, probs = model.detect_language(mels)
        return probs

    def _annotate(self, mmif: Mmif, **parameters) -> Mmif:
        window_ms = parameters['windowMs']
        if window_ms > MAX_WINDOW_MS:
            raise ValueError(f'windowMs must be at most {MAX_WINDOW_MS}, got {window_ms}')
        stride_ms = window_ms + parameters['windowGapMs']
        if stride_ms <= 0:
            raise ValueError(f'windowGapMs must be greater than -windowMs ({-window_ms})')
        model = self._get_model(parameters['model'])
        labelset = list(get_tokenizer(model.is_multilingual,
                                      num_languages=model.num_languages).all_language_codes)
        top_k = parameters['topK']

        for doc in chain(mmif.get_documents_by_type(DocumentTypes.AudioDocument),
                         mmif.get_documents_by_type(DocumentTypes.VideoDocument)):
            view = mmif.new_view()
            self.sign_view(view, parameters)
            view.new_contain(AnnotationTypes.TimeFrame, document=doc.id,
                             timeUnit='milliseconds', labelset=labelset)
            windows = adh.iter_audio_windows(
                doc, window_ms, stride_ms=stride_ms,
                sample_rate=whisper.audio.SAMPLE_RATE,
                min_last_window_ms=parameters['minLastWindowMs'])
            for batch in batched(windows, parameters['batchSize']):
                probs = self.detect(model, [samples for samples, _, _ in batch])
                for (_, start_ms, end_ms), window_probs in zip(batch, probs):
                    ranked = sorted(window_probs.items(), key=lambda kv: kv[1], reverse=True)
                    view.new_annotation(AnnotationTypes.TimeFrame,
                                        start=start_ms, end=end_ms, label=ranked[0][0],
                                        classification=dict(ranked[:top_k]))
        return mmif


def get_app():
    return WhisperSpokenLid()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", action="store", default="5000", help="set port to listen")
    parser.add_argument("--production", action="store_true", help="run gunicorn server")
    parsed_args = parser.parse_args()

    app = get_app()

    http_app = Restifier(app, port=int(parsed_args.port))
    if parsed_args.production:
        http_app.serve_production()
    else:
        app.logger.setLevel(logging.DEBUG)
        http_app.run()
