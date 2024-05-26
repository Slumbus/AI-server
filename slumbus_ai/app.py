from flask import Flask
from flask_restx import Api, Resource
from musicGen.musicGen import MusicGen

app = Flask(__name__) #FLASK 객체 선언, 애플리케이션 패키지 이름 넣기
api = Api(app) #Flask 객체에 APi 객체 등록

api.add_namespace(MusicGen, '/test')

@app.route('/')
def hello_pybo():
    return 'Hello, Pybo!'