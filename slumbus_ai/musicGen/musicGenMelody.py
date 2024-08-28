import tempfile
import uuid
from io import BytesIO

import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
from flask import request, jsonify
from flask_restx import Resource, Namespace

from .connection import s3_connection
from .config import BUCKET_NAME

from .spice import process_audio, upload_file

from pydub import AudioSegment

MusicGenMelody = Namespace('compose-music')

s3 = s3_connection()


@MusicGenMelody.route('/music', methods=['POST'])
class MusicComposer(Resource):
    def post(self):

        global fileUrl

        file = request.files['file']

        mood = request.form.get('mood')
        instrument = request.form.get('instrument')

        model = MusicGen.get_pretrained('melody')
        model.set_generation_params(duration=60)

        descriptions = [
            f"lullaby for a baby.The mood of the song should be {mood}.use {instrument} as main instrument. smooth, cozy and slow."
        ]

        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_file.write(file.read())
            temp_file.seek(0)

            spice_file = process_audio(temp_file)

            melody, sr = torchaudio.load(spice_file)  # 여기에 입력 받은 파일

            # 한 개의 채널로 음악 생성
            wav = model.generate_with_chroma(descriptions, melody[None], sr)

            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_output_file:
                torchaudio.save(temp_output_file, wav[0].cpu(), model.sample_rate, format="wav")

                temp_output_file.seek(0)
                fileName = f'music/{uuid.uuid4()}.wav'
                if uploadToS3(temp_output_file, BUCKET_NAME, fileName):
                    fileUrl = f'https://{BUCKET_NAME}.s3.ap-southeast-2.amazonaws.com/{fileName}'

        return jsonify({"music": fileUrl})


def uploadToS3(file, bucket, fileName):
    s3_client = s3_connection()
    try:
        s3_client.upload_fileobj(file, bucket, fileName, ExtraArgs={'ContentType': 'audio/wav'})
    except Exception as e:
        print(f"Error uploading file: {e}")
        return False
    return True
