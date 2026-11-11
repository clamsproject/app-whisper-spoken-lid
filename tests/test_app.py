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
# most probable language of each 1000 ms window of the test audio
TOPS = ['en', 'en', 'fr', 'en']


def probs_for(i):
    top = TOPS[i % len(TOPS)]
    rest = [lang for lang in ('en', 'fr', 'de') if lang != top]
    return {top: 0.6 + 0.1 * (i % 3), rest[0]: 0.1, rest[1]: 0.05}


class FakeModel:
    """Stands in for a Whisper model: records batch sizes, and returns the
    probabilities of ``probs_for`` in window order."""

    dims = SimpleNamespace(n_mels=80)
    device = 'cpu'
    is_multilingual = True
    num_languages = 99

    def __init__(self, probs_for=probs_for):
        self.probs_for = probs_for
        self.batch_sizes = []
        self.seen = 0

    def detect_language(self, mels):
        assert mels.shape[1:] == (80, 3000)
        self.batch_sizes.append(len(mels))
        probs = [self.probs_for(self.seen + i) for i in range(len(mels))]
        self.seen += len(mels)
        return None, probs


def use_model(monkeypatch, fake):
    monkeypatch.setattr(app.whisper, 'load_model', lambda size: fake)
    return fake


@pytest.fixture
def model(monkeypatch):
    return use_model(monkeypatch, FakeModel())


@pytest.fixture
def audio_path(tmp_path):
    path = tmp_path / 'tone.wav'
    t = np.arange(int(RATE * SECONDS)) / RATE
    samples = (np.sin(2 * np.pi * 440 * t) * 10000).astype('int16')[None, :]
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('pcm_s16le', rate=RATE, layout='mono')
        frame = av.AudioFrame.from_ndarray(samples, format='s16',
                                           layout='mono')
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
    params = {k: [str(x) for x in v] if isinstance(v, list) else [str(v)]
              for k, v in params.items()}
    return Mmif(app.get_app().annotate(mmif, **params))


def timeframes(out):
    view = out.get_view_contains(AnnotationTypes.TimeFrame)
    tfs = list(view.get_annotations(AnnotationTypes.TimeFrame))
    return view, [(tf.get_property('start'), tf.get_property('end'),
                   tf.get_property('label'), tf.get_property('scores'))
                  for tf in tfs]


def test_segments():
    windows = [(0, 10, 'en', .9), (10, 20, 'en', .8), (20, 25, 'fr', .7)]
    assert list(app._segments(windows)) == [
        (0, 20, 'en', [.9, .8]), (20, 25, 'fr', [.7])]
    assert list(app._segments(windows, merge=False)) == [
        (0, 10, 'en', [.9]), (10, 20, 'en', [.8]), (20, 25, 'fr', [.7])]
    # windows that do not touch stay apart
    assert list(app._segments([(0, 10, 'en', .9), (15, 20, 'en', .8)])) == [
        (0, 10, 'en', [.9]), (15, 20, 'en', [.8])]


def test_same_language_windows_are_merged(model, audio_path):
    view, tfs = timeframes(run(audio_path, windowMs=1000))
    # one score per window: the probability of the label in that window
    assert tfs == [(0, 2000, 'en', [0.6, 0.7]), (2000, 3000, 'fr', [0.8]),
                   (3000, 3500, 'en', [0.6])]
    contain = view.metadata.contains[AnnotationTypes.TimeFrame]
    assert contain['document'] == 'd1'
    assert contain['timeUnit'] == 'milliseconds'
    assert len(contain['labelset']) == FakeModel.num_languages
    # whisper's legacy code becomes a BCP 47 tag
    assert 'jv' in contain['labelset'] and 'jw' not in contain['labelset']


def test_legacy_codes_become_bcp47_tags(monkeypatch, audio_path):
    use_model(monkeypatch, FakeModel(lambda i: {'jw': 0.8, 'en': 0.2}))
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False))
    assert [label for _, _, label, _ in tfs] == ['jv'] * 4


def test_merge_off_gives_one_timeframe_per_window(model, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False))
    assert [(s, e) for s, e, _, _ in tfs] == [
        (0, 1000), (1000, 2000), (2000, 3000), (3000, 3500)]
    assert all(len(scores) == 1 for _, _, _, scores in tfs)


def test_short_last_window_is_dropped(model, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, minLastWindowMs=600))
    assert tfs[-1][:3] == (2000, 3000, 'fr')


def test_batches_follow_batch_size(model, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, batchSize=3))
    assert model.batch_sizes == [3, 1]
    assert len(tfs) == 3


def test_window_longer_than_whisper_input_raises(model, audio_path):
    with pytest.raises(ValueError, match='at most 30000'):
        run(audio_path, windowMs=30001)


# probabilities over all labels for each 1000 ms window
PROBS = [{'en': .6, 'es': .1, 'fr': .1, 'de': .1, 'jw': .1},
         {'en': .3, 'es': .3, 'fr': .4}]


@pytest.fixture
def pooled(monkeypatch):
    return use_model(monkeypatch,
                     FakeModel(lambda i: dict(PROBS[i % len(PROBS)])))


def test_parse_label_map():
    labelset = ['en', 'es', 'fr']
    assert app._parse_label_map(['en', 'es:spanish'], labelset) == {
        'en': 'en', 'es': 'spanish'}
    with pytest.raises(ValueError, match='"xx" is not a label'):
        app._parse_label_map(['xx:en'], labelset)


def test_label_map_pools_probabilities(pooled, audio_path):
    view, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                               labelMap='en'))
    # window 2: en 0.3 against 0.7 for all other labels
    assert [(label, round(scores[0], 3)) for _, _, label, scores in tfs] == [
        ('en', 0.6), ('-', 0.7), ('en', 0.6), ('-', 0.7)]
    contain = view.metadata.contains[AnnotationTypes.TimeFrame]
    assert contain['labelset'] == ['en', '-']


def test_labels_mapped_together_add_up(pooled, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap=['en:target', 'es:target']))
    assert [(label, round(scores[0], 3)) for _, _, label, scores in tfs] == [
        ('target', 0.7), ('target', 0.6), ('target', 0.7), ('target', 0.6)]


def test_label_map_takes_bcp47_tags(pooled, audio_path):
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap=['jv', 'en'], minLabelScore=0.05))
    assert [label for _, _, label, _ in tfs] == ['en', 'en', 'en', 'en']
    with pytest.raises(ValueError, match='"jw" is not a label'):
        run(audio_path, windowMs=1000, labelMap='jw')


def test_min_label_score_sets_the_bar(pooled, audio_path):
    # window 2: en 0.3, below the default 0.5 but above 0.25
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap='en', minLabelScore=0.25))
    assert [(label, round(scores[0], 3)) for _, _, label, scores in tfs] == [
        ('en', 0.6), ('en', 0.3), ('en', 0.6), ('en', 0.3)]
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap='en', minLabelScore=0.7))
    assert [label for _, _, label, _ in tfs] == ['-'] * 4


def test_separate_labels_each_need_the_score(pooled, audio_path):
    # window 2: en 0.3 and es 0.3, each below 0.5
    _, tfs = timeframes(run(audio_path, windowMs=1000, tfMerge=False,
                            labelMap=['en', 'es']))
    assert [(label, round(scores[0], 3)) for _, _, label, scores in tfs] == [
        ('en', 0.6), ('-', 0.4), ('en', 0.6), ('-', 0.4)]
