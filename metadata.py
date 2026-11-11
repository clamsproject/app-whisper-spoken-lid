"""
The purpose of this file is to define the metadata of the app with minimal imports.

DO NOT CHANGE the name of the file
"""
import pathlib
import re

from clams.app import ClamsApp
from clams.appmetadata import AppMetadata
from mmif import DocumentTypes, AnnotationTypes

WHISPER_VERSION = [
    line.strip().rsplit('==')[-1]
    for line in open(pathlib.Path(__file__).parent / 'requirements.txt')
    if re.match(r'^openai-whisper==', line)][0]
MODEL_SIZES = ['tiny', 'base', 'small', 'medium', 'large', 'turbo']
# whisper pads or trims every input to 30 s
MAX_WINDOW_MS = 30000


def appmetadata() -> AppMetadata:
    metadata = AppMetadata(
        name="Whisper Spoken LID",
        description=(
            "Spoken language identification over fixed-length windows of "
            "an audio track, using the language detection of OpenAI "
            "Whisper. Consecutive windows with the same language can be "
            "merged into one TimeFrame."),
        app_license="Apache 2.0",
        identifier="whisper-spoken-lid",
        url="https://github.com/clamsproject/app-whisper-spoken-lid",
        analyzer_version=WHISPER_VERSION,
        analyzer_license="MIT",
        # approximate VRAM (MB) of the default "tiny" model; larger models
        # need more ("turbo" about 6 GB, "large" about 10 GB)
        est_gpu_mem_min=1500,
        est_gpu_mem_typ=1500,
    )
    metadata.add_input_oneof(DocumentTypes.AudioDocument,
                             DocumentTypes.VideoDocument)
    out_tf = metadata.add_output(AnnotationTypes.TimeFrame,
                                 timeUnit='milliseconds', document='*',
                                 labelset='*', label='*', scores='*')
    out_tf.add_description(
        'Each TimeFrame covers consecutive windows that Whisper labels with '
        'the language `label`. Language labels are canonicalized to BCP 47 '
        'tags, and `labelMap` can reduce them; see README.md in the app '
        'repository for details. `labelset` lists all labels the view can '
        'have. `scores` lists, in time order, the Whisper probability of '
        '`label` in each of those windows, one float in [0, 1] per window. '
        'Window i starts at `start` + i * `windowMs`; the last window ends '
        'at `end` and can be shorter. The probabilities are not '
        'calibrated.')

    metadata.add_parameter(
        name='model', type='string', default='tiny', choices=MODEL_SIZES,
        description='Whisper model size.')
    metadata.add_parameter(
        name='windowMs', type='integer', default=MAX_WINDOW_MS,
        description=f'Window length in milliseconds, at most '
                    f'{MAX_WINDOW_MS}.')
    metadata.add_parameter(
        name='minLastWindowMs', type='integer', default=0,
        description='Shortest final window to keep, in milliseconds. 0 '
                    'keeps any final window.')
    metadata.add_parameter(
        name='tfMerge', type='boolean', default=True,
        description='Join consecutive windows with the same label into one '
                    'TimeFrame. When false, each window becomes its own '
                    'TimeFrame, with one score.')
    metadata.add_parameter(
        name='labelMap', type='string', default=[], multivalued=True,
        description='Labels to keep, as "IN:OUT" (rename IN to OUT) or "IN" '
                    '(keep IN as it is). Other labels become "-". Empty '
                    'keeps every label of the model. See README.md in the '
                    'app repository for how the label is chosen.')
    metadata.add_parameter(
        name='minLabelScore', type='number', default=0.5,
        description='Minimum score for a label from labelMap to be kept; a '
                    'window whose best mapped label scores less becomes '
                    '"-". Only applies with labelMap.')
    metadata.add_parameter(
        name='batchSize', type='integer', default=1,
        description='Number of windows passed to the model at once.')
    return metadata


# DO NOT CHANGE the main block
if __name__ == '__main__':
    import sys
    metadata = appmetadata()
    for param in ClamsApp.universal_parameters:
        metadata.add_parameter(**param)
    sys.stdout.write(metadata.jsonify(pretty=True))
