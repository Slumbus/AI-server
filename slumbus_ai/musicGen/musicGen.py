from flask_restx import Resource, Api, Namespace


from transformers import pipeline
import scipy

from transformers import AutoProcessor, MusicgenForConditionalGeneration

from IPython.display import Audio

import scipy

MusicGen = Namespace('generate-music')

@MusicGen.route('/music')
class MusicGenerator(Resource):
    def get(self):
        ##1
        # 파이프 라인 생성 -> text to audio 작업 수행
        # synthesiser = pipeline("text-to-audio", "facebook/musicgen-medium")
        # # 해당 inputs: 텍스트를 바탕으로 오디오 생성 , do_sample:True -> 샘플링 활성화로 무작위성 확보
        # music = synthesiser("lo-fi music with a soothing melody", forward_params={"do_sample": True})
        # # 생성된 오디오를 저장 rate: 생성된 오디오의 샘플링 속도, data: 오디오 데이터 전달 (모두 music 객체에서 가져옴)
        # scipy.io.wavfile.write("musicgen_out.wav", rate=music["sampling_rate"], data=music["audio"])

        ## 2
        # 프로세서 및 모델 로드
        processor = AutoProcessor.from_pretrained("facebook/musicgen-medium")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")

        # 입력 텍스트 처리
        inputs = processor(
            text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums"],
            padding=True,
            return_tensors="pt",
        )

        # 오디오 생성
        audio_values = model.generate(**inputs, max_new_tokens=256)

        ## 2-1
        # IPython 노트북에서 생성된 오디오 데이터를 재생할 수 있는 위젯 제공하는 코드
        # 샘플링 속도 가져오기
        # sampling_rate = model.config.audio_encoder.sampling_rate
        # # 오디오 재생
        # Audio(audio_values[0].numpy(), rate=sampling_rate)

        ## 2-2
        # 샘플링 속도 가져오기
        sampling_rate = model.config.audio_encoder.sampling_rate
        # 오디오 저장 -> 아직 디렉토리 지정 안함.
        scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())


