from flask import request, jsonify
from flask_restx import Resource, Api, Namespace

import openai
import os

LyricsWriter = Namespace('write-lyrics')

@LyricsWriter.route('/write')
class WriteLyrics(Resource):

    def post(self):
        input = request.get_json()

        lyrics = input["lyrics"]

        openai.api_key = os.environ["GPT_API_KEY"]

        completion = openai.chat.completions.create(
            model="gpt-4o",
            messages = [
                {"role": "system", "content": "You are a lyricist for children's lullaby."},
                {"role": "system", "content": "Write a lullaby with four stanzas, each consisting of 3 to 4 lines."},
                {"role": "system", "content": "Write the lyrics according to the user's request."},
                {"role": "system", "content": "The lyrics must be in Korean."},
                {"role": "system", "content": "Just return the lyrics of the lullaby you wrote."},
                {"role": "system", "content": "The lyrics are for a lullaby for a child."},
                {"role": "system", "content": "If there is no request, just return the Korean lullaby lyrics"},
                {"role": "system", "content": "Do not include any special characters or symbols in the lyrics."},
                {"role": "system", "content": "do not include the child's name in the lyrics if it is unknown."},
                {"role": "user", "content": f"${lyrics} is my request. reflect this request and write the lyrics"},
            ],
            temperature=1.2,
            max_tokens=2048
        )

        message_result = completion.choices[0].message.content.encode("utf-8").decode()

        return jsonify({"result": message_result})