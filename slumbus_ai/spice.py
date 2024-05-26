import logging
import math
import statistics

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from scipy.io import wavfile
from pydub import AudioSegment
import music21

# Logger 설정
logger = logging.getLogger()
logger.setLevel(logging.ERROR)

# Constants
EXPECTED_SAMPLE_RATE = 16000
MAX_ABS_INT16 = 32768.0
A4 = 440
C0 = A4 * pow(2, -4.75)
note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# 모델 로드
model = hub.load("https://tfhub.dev/google/spice/2")


def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
    audio = AudioSegment.from_file(user_file)
    audio = audio.set_frame_rate(EXPECTED_SAMPLE_RATE).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file


def output2hz(pitch_output):
    PT_OFFSET = 25.58
    PT_SLOPE = 63.07
    FMIN = 10.0
    BINS_PER_OCTAVE = 12.0
    cqt_bin = pitch_output * PT_SLOPE + PT_OFFSET
    return FMIN * 2.0 ** (1.0 * cqt_bin / BINS_PER_OCTAVE)


def hz2offset(freq):
    if freq == 0:
        return None
    h = round(12 * math.log2(freq / C0))
    return 12 * math.log2(freq / C0) - h


def quantize_predictions(group, ideal_offset):
    non_zero_values = [v for v in group if v != 0]
    zero_values_count = len(group) - len(non_zero_values)

    if zero_values_count > 0.8 * len(group):
        return 0.51 * len(non_zero_values), "Rest"
    else:
        h = round(statistics.mean([12 * math.log2(freq / C0) - ideal_offset for freq in non_zero_values]))
        octave = h // 12
        n = h % 12
        note = note_names[n] + str(octave)
        error = sum([abs(12 * math.log2(freq / C0) - ideal_offset - h) for freq in non_zero_values])
        return error, note


def get_quantization_and_error(pitch_outputs_and_rests, predictions_per_eighth, prediction_start_offset, ideal_offset):
    pitch_outputs_and_rests = [0] * prediction_start_offset + pitch_outputs_and_rests
    groups = [pitch_outputs_and_rests[i:i + predictions_per_eighth] for i in
              range(0, len(pitch_outputs_and_rests), predictions_per_eighth)]

    quantization_error = 0
    notes_and_rests = []

    for group in groups:
        error, note_or_rest = quantize_predictions(group, ideal_offset)
        quantization_error += error
        notes_and_rests.append(note_or_rest)

    return quantization_error, notes_and_rests


def process_audio(file_path):
    converted_audio_file = convert_audio_for_model(file_path)
    sample_rate, audio_samples = wavfile.read(converted_audio_file, 'rb')

    audio_samples = audio_samples / float(MAX_ABS_INT16)
    model_output = model.signatures["serving_default"](tf.constant(audio_samples, tf.float32))

    pitch_outputs = model_output["pitch"]
    uncertainty_outputs = model_output["uncertainty"]
    confidence_outputs = 1.0 - uncertainty_outputs

    pitch_outputs = [float(x) for x in pitch_outputs]
    confidence_outputs = list(confidence_outputs)

    indices = range(len(pitch_outputs))
    confident_pitch_outputs = [(i, p) for i, p, c in zip(indices, pitch_outputs, confidence_outputs) if c >= 0.9]
    confident_pitch_outputs_x, confident_pitch_outputs_y = zip(*confident_pitch_outputs)

    confident_pitch_values_hz = [output2hz(p) for p in confident_pitch_outputs_y]

    pitch_outputs_and_rests = [output2hz(p) if c >= 0.9 else 0 for i, p, c in
                               zip(indices, pitch_outputs, confidence_outputs)]
    offsets = [hz2offset(p) for p in pitch_outputs_and_rests if p != 0]

    ideal_offset = statistics.mean(offsets)

    best_error = float("inf")
    best_notes_and_rests = None
    best_predictions_per_note = None

    for predictions_per_note in range(20, 65, 1):
        for prediction_start_offset in range(predictions_per_note):
            error, notes_and_rests = get_quantization_and_error(pitch_outputs_and_rests, predictions_per_note,
                                                                prediction_start_offset, ideal_offset)
            if error < best_error:
                best_error = error
                best_notes_and_rests = notes_and_rests
                best_predictions_per_note = predictions_per_note

    while best_notes_and_rests[0] == 'Rest':
        best_notes_and_rests = best_notes_and_rests[1:]
    while best_notes_and_rests[-1] == 'Rest':
        best_notes_and_rests = best_notes_and_rests[:-1]

    sc = music21.stream.Score()
    bpm = 60 * 60 / best_predictions_per_note
    a = music21.tempo.MetronomeMark(number=bpm)
    sc.insert(0, a)

    for snote in best_notes_and_rests:
        d = 'half'
        if snote == 'Rest':
            sc.append(music21.note.Rest(type=d))
        else:
            sc.append(music21.note.Note(snote, type=d))

    midi_file = file_path[:-4] + '.mid'
    sc.write('midi', fp=midi_file)

    return midi_file
