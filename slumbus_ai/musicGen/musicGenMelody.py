import uuid
from io import BytesIO

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from flask import request, jsonify
from flask_restx import Resource, Namespace

from .connection import s3_connection
from .config import BUCKET_NAME

from pydub import AudioSegment

MusicGenMelody = Namespace('compose-music')

s3 = s3_connection()
@MusicGenMelody.route('/music')
class MusicComposer(Resource):
    def post(self):

        global fileUrl

        file = request.files['file']
        file_stream = BytesIO(file.read())

        mood = request.form.get('mood')
        instrument = request.form.get('instrument')

        model = MusicGen.get_pretrained('melody')
        model.set_generation_params(duration=60)  # generate 8 seconds.

        descriptions = [
            f"music style is {mood} and instruments are {instrument}"
        ]

        # wav 파일이 아니라면 변환해주기
        # output_file = convert_audio_for_model(file_stream)

        melody, sr = torchaudio.load(file_stream) # 여기에 입력 받은 파일

        # 한 개의 채널로 음악 생성
        wav = model.generate_with_chroma(descriptions, melody[None],sr)
        # 세 개의 채널의 음악 생성
        # wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr) # 1:mono, 2:stereo, 3:multichannel

        for idx, one_wav in enumerate(wav):
            # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
            buffer = BytesIO()
            torchaudio.save(buffer, one_wav.cpu(), model.sample_rate, format="wav")
            # audio_write(buffer, one_wav.cpu(), model.sample_rate, strategy="loudness")
            buffer.seek(0)

            fileName = f'music/{uuid.uuid4()}.wav'
            if uploadToS3(buffer, BUCKET_NAME, fileName):
                fileUrl = f'https://{BUCKET_NAME}.s3.ap-southeast-2.amazonaws.com/{fileName}'

        return jsonify({"music": fileUrl})

def convert_audio_for_model(user_file, output_file='converted_audio_file.wav'):
    audio = AudioSegment.from_file(user_file)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(output_file, format="wav")
    return output_file

def uploadToS3(file, bucket, fileName):
    s3_client = s3_connection()
    try:
        s3_client.upload_fileobj(file, bucket, fileName, ExtraArgs={'ContentType': 'audio/wav'})
    except Exception as e:
        print(f"Error uploading file: {e}")
        return False
    return True