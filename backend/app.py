"""
app.py
------
AgriVision AI — Flask backend.

Routes:
  GET  /            Home page
  GET  /detect       Detection page (upload UI)
  POST /api/predict  Runs the pretrained model on an uploaded image, returns JSON
  GET  /about        About the project page

Run:
  python backend/app.py
Then open http://127.0.0.1:5000
"""
import os
import sys
import uuid

from flask import Flask, jsonify, render_template, request
from PIL import Image
from werkzeug.utils import secure_filename

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.utils import is_model_available, predict
from training.config import UPLOAD_DIR

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "templates"),
    static_folder=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend", "static"),
)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_DIR, exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html", model_ready=is_model_available())


@app.route("/detect")
def detect_page():
    return render_template("detect.html", model_ready=is_model_available())


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/api/predict", methods=["POST"])
def api_predict():
    if not is_model_available():
        return jsonify({
            "error": (
                "Required AI libraries (torch, transformers) are not installed. "
                "Run: pip install -r requirements.txt, then restart the server."
            )
        }), 503

    if "image" not in request.files:
        return jsonify({"error": "No image file provided (expected form field 'image')."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Unsupported file type. Please upload a JPG or PNG image."}), 400

    try:
        pil_image = Image.open(file.stream)
        pil_image.verify()  # sanity-check it's a real image
        file.stream.seek(0)
        pil_image = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "The uploaded file is not a valid image."}), 400

    # Save a copy (useful for the demo / result page, and for debugging)
    safe_name = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    saved_path = os.path.join(UPLOAD_DIR, safe_name)
    pil_image.save(saved_path)

    try:
        result = predict(pil_image)
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

    result["image_url"] = f"/static/uploads/{safe_name}"
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
