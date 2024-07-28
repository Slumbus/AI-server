from flask import Flask, request, jsonify, send_file, render_template
from spice import process_audio

from flask_restx import Api, Resource
from musicGen.musicGen import MusicGen
from musicGen.musicGenMelody import MusicGenMelody
from chatGPT.chatGpt import LyricsWriter

app = Flask(__name__) #FLASK 객체 선언, 애플리케이션 패키지 이름 넣기
api = Api(app) #Flask 객체에 APi 객체 등록

api.add_namespace(MusicGen, '/test')
api.add_namespace(MusicGenMelody, '/compose')
api.add_namespace(LyricsWriter, '/lyrics')

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify(error="No file part"), 400

    file = request.files['file']
    # If the user does not select a file, the browser submits an empty file without a filename
    if file.filename == '':
        return jsonify(error="No selected file"), 400

    # If the file exists and is valid
    if file:
        # Save the file to the server
        filename = 'recorded_audio.wav'
        file.save(filename)

        # Process the audio file
        midi_file = process_audio(filename)

        # Return the processed MIDI file to the user
        return send_file(midi_file, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
