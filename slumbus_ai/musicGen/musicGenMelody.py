from io import BytesIO

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from flask import request, jsonify
from flask_restx import Resource, Api, Namespace
import os

from pydub import AudioSegment

MusicGenMelody = Namespace('compose-music')

@MusicGenMelody.route('/music')
class MusicComposer(Resource):
    def post(self):

        file = request.files['file']
        file_stream = BytesIO(file.read())

        mood = request.form.get('mood')
        instrument = request.form.get('instrument')

        model = MusicGen.get_pretrained('melody')
        model.set_generation_params(duration=60)  # generate 8 seconds.

        descriptions = [
            f"music style is {mood} and instruments are {instrument}"
        ]

        output_file = convert_audio_for_model(file_stream)

        melody, sr = torchaudio.load(output_file) # 여기에 입력 받은 파일
        # generates using the melody from the given audio and the provided descriptions.
        # 한 개의 채널로 음악 생성
        wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1),sr)
        # wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr) # 1:mono, 2:stereo, 3:multichannel

        for idx, one_wav in enumerate(wav):
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness")



def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
    audio = AudioSegment.from_file(user_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file

