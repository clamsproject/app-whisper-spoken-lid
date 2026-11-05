import sys
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
from mmif import AnnotationTypes, Document, DocumentTypes, Mmif

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app  # noqa: E402

RATE = 16000
SECONDS = 3.5
PROBS = {'en': 0.7, 'fr': 0.2, 'de': 0.06, 'es': 0.04}


class FakeModel:
    """Stands in for a Whisper model: records batch sizes, returns PROBS."""

    dims = SimpleNamespace(n_mels=80)
    device = 'cpu'
    is_multilingual = True
    num_languages = 99

    def __init__(self):
        self.batch_sizes = []

    def detect_language(self, mels):
        assert mels.shape[1:] == (80, 3000)
        self.batch_sizes.append(len(mels))
        return None, [dict(PROBS) for _ in range(len(mels))]


@pytest.fixture
def model(monkeypatch):
    fake = FakeModel()
    monkeypatch.setattr(app.whisper, 'load_model', lambda size: fake)
    return fake


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / 'tone.wav'
    t = np.arange(int(RATE * SECONDS)) / RATE
    samples = (np.sin(2 * np.pi * 440 * t) * 10000).astype('int16')[None, :]
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('pcm_s16le', rate=RATE, layout='mono')
        frame = av.AudioFrame.from_ndarray(samples, format='s16', layout='mono')
        frame.sample_rate = RATE
        frame.pts = 0
        frame.time_base = Fraction(1, RATE)
        out.mux(stream.encode(frame))
        out.mux(stream.encode(None))
    return path


def run(audio_path, **params):
    mmif = Mmif(validate=False)
    mmif.add_document(Document({
        '@type': DocumentTypes.AudioDocument,
        'properties': {'id': 'd1', 'mime': 'audio/wav',
                       'location': f'file://{audio_path}'}}))
    params = {k: [str(v)] for k, v in params.items()}
    return Mmif(app.get_app().annotate(mmif, **params))


def timeframes(out):
    view = out.views[0]
    return view, list(view.get_annotations(AnnotationTypes.TimeFrame))


def spans(tfs):
    return [(tf.get_property('start'), tf.get_property('end')) for tf in tfs]


def test_windows_become_labeled_timeframes(model, audio_path):
    view, tfs = timeframes(run(audio_path, windowMs=1000, topK=2))
    assert spans(tfs) == [(0, 1000), (1000, 2000), (2000, 3000), (3000, 3500)]
    assert all(tf.get_property('label') == 'en' for tf in tfs)
    assert tfs[0].get_property('classification') == {'en': 0.7, 'fr': 0.2}
    contain = view.metadata.contains[AnnotationTypes.TimeFrame]
    assert contain['document'] == 'd1'
    assert contain['timeUnit'] == 'milliseconds'
    assert len(contain['labelset']) == FakeModel.num_languages


def test_window_gap_and_last_window(model, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=2000, windowGapMs=-1000,
                            minLastWindowMs=1000))
    assert spans(tfs) == [(0, 2000), (1000, 3000), (2000, 3500)]
    _, tfs = timeframes(run(audio_path, windowMs=1000, windowGapMs=500))
    assert spans(tfs) == [(0, 1000), (1500, 2500), (3000, 3500)]
    with pytest.raises(ValueError, match='windowGapMs'):
        run(audio_path, windowMs=1000, windowGapMs=-1000)


def test_batches_follow_batch_size(model, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, batchSize=3))
    assert model.batch_sizes == [3, 1]
    assert len(tfs) == 4


def test_window_longer_than_whisper_input_raises(model, audio_path):
    with pytest.raises(ValueError, match='at most 30000'):
        run(audio_path, windowMs=30001)
