from flask import request, jsonify
from flask_restx import Resource, Api, Namespace

from transformers import AutoProcessor, MusicgenForConditionalGeneration

import scipy
import os

MusicGen = Namespace('generate-music')

@MusicGen.route('/music')
class MusicGenerator(Resource):
    def post(self):

        input = request.get_json()

        mood = input["mood"]
        instrument = input["instrument"]

        # 프로세서 및 모델 로드
        processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")

        # 입력 텍스트 처리
        inputs = processor(
            text=[f" 60초 짜리 자장가 음악을 만들어 줘. 분위기는 ${mood}이고, 악기는 ${instrument} 이걸 사용해 줘."],
            padding=True,
            return_tensors="pt",
        )

        # 오디오 생성
        audio_values = model.generate(**inputs, max_new_tokens=256)


        ## 2-2
        # 샘플링 속도 가져오기
        sampling_rate = model.config.audio_encoder.sampling_rate
        # 오디오 저장
        scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())


