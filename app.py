import os
from flask import Flask, request, jsonify
import assemblyai as aai
import requests

app = Flask(__name__)

# Replace with your AssemblyAI API key
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json()
    file_url = data.get('file_url')
    
    if not file_url:
        print("No file URL provided")
        return jsonify({"error": "File URL is required"}), 400

    print(f"File URL received: {file_url}")
    
    try:
        # Verify the uploaded file is indeed an audio file
        head_response = requests.head(file_url)
        if 'audio' not in head_response.headers.get('Content-Type', ''):
            print(f"Uploaded file is not recognized as audio: {head_response.headers.get('Content-Type')}")
            return jsonify({"error": "Uploaded file is not recognized as audio"}), 400
        
        # Configure the transcription with speaker labels and auto-highlights
        config = aai.TranscriptionConfig(speaker_labels=True, auto_highlights=True)
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_url, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"Transcription error: {transcript.error}")
            return jsonify({"error": transcript.error}), 500

        response = {
            "text": transcript.text,
            "speakers": [{"speaker": u.speaker, "text": u.text} for u in (transcript.utterances or [])],
            "highlights": [{"text": h.text, "count": h.count, "rank": h.rank} for h in (getattr(transcript.auto_highlights, 'results', []) or [])]
        }
        print(f"Transcription successful: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Exception: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
