from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import whisper
import requests
import json
import socket


def diagnose_lm_studio_connection():
    diagnostics = {
        "network_checks": {},
        "endpoint_checks": {}
    }
    test_endpoints = ["localhost", "127.0.0.1", "192.168.0.101"]
    test_ports = [1234, 11434, 8000]
    for endpoint in test_endpoints:
        for port in test_ports:
            key = f"{endpoint}:{port}"
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((endpoint, port))
                diagnostics["network_checks"][key] = "Open" if result == 0 else "Closed"
                sock.close()
            except Exception as e:
                diagnostics["network_checks"][key] = f"Error: {str(e)}"

    test_urls = [
        "http://localhost:1234",
        "http://localhost:1234/v1/chat/completions",
        "http://localhost:1234/chat",
        "http://127.0.0.1:1234",
        "http://192.168.0.101:1234"
    ]
    for url in test_urls:
        try:
            response = requests.get(url, timeout=3)
            diagnostics["endpoint_checks"][url] = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.text[:200]
            }
        except requests.ConnectionError:
            diagnostics["endpoint_checks"][url] = "Connection Error"
        except requests.Timeout:
            diagnostics["endpoint_checks"][url] = "Timeout"
        except Exception as e:
            diagnostics["endpoint_checks"][url] = f"Error: {str(e)}"
    return diagnostics


app = Flask(__name__)
app.secret_key = "my-dev-secret-key-123"
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

model = whisper.load_model("base")


@app.route("/")
def index():
    return render_template(
        "index.html",
        transcript=session.get("transcript", ""),
        action_points=session.get("action_points", "")
    )


def has_audio_stream(filepath):
    """Check if the video file contains an audio stream using ffprobe."""
    import subprocess
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'a',
            '-show_entries', 'stream=codec_type',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            filepath
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return 'audio' in result.stdout.lower()
    except Exception as e:
        print(f"Error checking audio stream: {e}")
        return False

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return "No file uploaded", 400
    
    file = request.files["file"]
    mode = request.form.get("mode", "script")
    
    if file.filename == "":
        return "No selected file", 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    # Check if file has audio stream
    if not has_audio_stream(filepath):
        return "Error: The selected video file does not contain any audio stream. Please upload a video with audio.", 400
    
    try:
        result = model.transcribe(filepath)
        full_transcript = result["text"]
        session["transcript"] = full_transcript
        session["mode"] = mode
        return redirect(url_for("chat"))
    except Exception as e:
        return f"Error during transcription: {str(e)}", 500


@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.args.get('diagnose') == 'true':
        diagnostics = diagnose_lm_studio_connection()
        return jsonify(diagnostics)

    if request.method == 'POST':
        prompt = request.form.get('prompt', '')
        if not prompt:
            return jsonify({"response": "No prompt provided."})

        import sys

        print(f"Received prompt: {prompt}", file=sys.stderr)  # 👈 PLACE IT HERE

        lmstudio_url = "http://192.168.0.101:1234/v1/completions"
        headers = {"Content-Type": "application/json"}

        payload = {
            "model": "meta-llama-3.1-8b-instruct",
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7
        }

        try:
            # Diagnostic print to help troubleshoot connection
            print(f"Attempting to connect to: {lmstudio_url}", file=sys.stderr)
            print(f"Payload: {json.dumps(payload)}", file=sys.stderr)

            response = requests.post(lmstudio_url, json=payload, headers=headers, timeout=50)
            print(f"Response Status: {response.status_code}", file=sys.stderr)
            print(f"Response Headers: {response.headers}", file=sys.stderr)
            print(f"Response Text: {response.text[:500]}", file=sys.stderr)  # Limit text to first 500 chars

            if response.status_code == 200:
                try:
                    response_json = response.json()

                    # More robust JSON parsing
                    answer = response_json.get("choices", [{}])[0].get("text", "").strip()

                    if not answer:
                        answer = "[No response content from model]"

                    return jsonify({"response": answer})
                except json.JSONDecodeError as je:
                    print(f"JSON Decode Error: {je}", file=sys.stderr)
                    return jsonify({"response": f"[JSON Parsing Error] {str(je)}"})

            else:
                return jsonify({"response": f"[LM Studio Error] {response.status_code}: {response.text[:200]}"})

        except requests.ConnectionError as ce:
            print(f"Connection Error: {ce}", file=sys.stderr)
            return jsonify({"response": f"[Connection Error] Unable to reach LM Studio server. Check server status."})
        except requests.Timeout as te:
            print(f"Connection Timeout: {te}", file=sys.stderr)
            return jsonify({"response": f"[Timeout Error] LM Studio server took too long to respond."})
        except requests.RequestException as e:
            print(f"Request Exception: {e}", file=sys.stderr)
            return jsonify({"response": f"[Request Exception] {str(e)}"})

    return render_template("chat.html")



if __name__ == "__main__":
    app.run(debug=True)
