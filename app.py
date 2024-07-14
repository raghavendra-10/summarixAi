import os
from flask import Flask, request, jsonify
import assemblyai as aai
import requests
import cohere
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Use the AssemblyAI API key from the environment variable
aai.settings.api_key = os.getenv('ASSEMBLYAI_API_KEY')

# Initialize the Cohere client with the API key from the environment variable
cohere_api_key = os.getenv('COHERE_API_KEY')
co = cohere.Client(cohere_api_key)

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
        
        # Configure the transcription with enhanced speaker labels and auto-highlights
        config = aai.TranscriptionConfig(speaker_labels=True, auto_highlights=True)
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(file_url, config=config)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"Transcription error: {transcript.error}")
            return jsonify({"error": transcript.error}), 500

        # Prepare the transcription data
        response = {
            "text": transcript.text,
            "speakers": [{"speaker": u.speaker, "text": u.text} for u in (transcript.utterances or [])],
            "highlights": [{"text": h.text, "count": h.count, "rank": h.rank} for h in (getattr(transcript.auto_highlights, 'results', []) or [])]
        }

        # Use Cohere's generation model to summarize the transcription
        summary = summarize_text(response["text"])
        response["summary"] = summary
        print(f"Summarization successful: {response}")
        return jsonify(response)

    except Exception as e:
        print(f"Exception: {e}")
        return jsonify({"error": str(e)}), 500

def summarize_text(text):
    try:
        prompt = f"Summarize the following text in a concise manner for:\n\n{text}\n\nSummary:"
        response = co.generate(
            model='command-r-plus',
            prompt=prompt,
            max_tokens=200,
            temperature=0.7,
        )
        summary = response.generations[0].text.strip()
        print(f"Summarization successful: {summary}")
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        return "Error during summarization."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
