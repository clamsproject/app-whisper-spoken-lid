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
        description="Spoken language identification over fixed-length windows of an audio track, "
                    "using the language detection of OpenAI Whisper. Each window becomes a "
                    "TimeFrame labeled with the most probable language code.",
        app_license="Apache 2.0",
        identifier="whisper-spoken-lid",
        url="https://github.com/clamsproject/app-whisper-spoken-lid",
        analyzer_version=WHISPER_VERSION,
        analyzer_license="MIT",
    )
    metadata.add_input_oneof(DocumentTypes.AudioDocument, DocumentTypes.VideoDocument)
    metadata.add_output(AnnotationTypes.TimeFrame, timeUnit='milliseconds')

    metadata.add_parameter(
        name='model', type='string', default='tiny', choices=MODEL_SIZES,
        description='Whisper model size.')
    metadata.add_parameter(
        name='windowMs', type='integer', default=MAX_WINDOW_MS,
        description=f'Window length in milliseconds, at most {MAX_WINDOW_MS}.')
    metadata.add_parameter(
        name='windowGapMs', type='integer', default=0,
        description='Milliseconds from the end of one window to the start of the next. '
                    'A negative value makes windows overlap.')
    metadata.add_parameter(
        name='minLastWindowMs', type='integer', default=0,
        description='Shortest final window to keep, in milliseconds. 0 keeps any final window.')
    metadata.add_parameter(
        name='topK', type='integer', default=3,
        description='Number of most probable languages kept in the `classification` property.')
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
