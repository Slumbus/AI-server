from flask import request, jsonify
from flask_restx import Resource, Api, Namespace

import openai
import os

LyricsWriter = Namespace('write-lyrics')

@LyricsWriter.route('/write')
class WriteLyrics(Resource):

    def post(self):
        input = request.get_json()

        kidName = input["kidName"]

        openai.api_key = os.environ["GPT_API_KEY"]

        completion = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages = [
                {"role": "system", "content": "너는 작사가이다."},
                {"role": "system", "content": "사용자가 요청하는 조건에 따라 작사를 해라."},
                {"role": "system", "content": "작사된 가사만 반환해라."},
                {"role": "system", "content": "작사할 내용은 자장가의 가사이다."},
                {"role": "user", "content": f"${kidName}이 아이의 이름이야. 아이의 이름이 들어가도록 작사해줘."},
            ],
            temperature=0.8,
            max_tokens=2048
        )

        message_result = completion.choices[0].message.content.encode("utf-8").decode()

        return jsonify({"result": message_result})