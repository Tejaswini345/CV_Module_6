"""
app.py - web demo.   Local:  python app.py   ->  http://127.0.0.1:5000
Server: gunicorn app:app
Videos in videos/ appear in the dropdown. Flow results are cached in outputs/.
"""
import os, json
from flask import Flask, jsonify, request, render_template, send_from_directory
import flow, sfm

app = Flask(__name__)
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")
os.makedirs("outputs", exist_ok=True)


@app.route("/")
def index():
    vids = sorted(f for f in os.listdir("videos") if f.lower().endswith(VIDEO_EXT))
    return render_template("index.html", videos=vids)


@app.route("/videos/<path:f>")
def raw_video(f):
    return send_from_directory("videos", f)


@app.route("/outputs/<path:f>")
def outputs(f):
    return send_from_directory("outputs", f)


def _name(v):
    return os.path.splitext(v)[0]


@app.route("/api/flow", methods=["POST"])
def api_flow():
    v = request.json["video"]
    name = _name(v)
    cache = f"outputs/{name}_flowcache.json"
    if os.path.exists(cache) and os.path.exists(f"outputs/{name}_flow.mp4"):
        return jsonify(json.load(open(cache)))           # saved result: instant
    result = flow.dense_flow_video(os.path.join("videos", v), name)
    json.dump(result, open(cache, "w"))
    return jsonify(result)


@app.route("/api/track", methods=["POST"])
def api_track():
    j = request.json
    try:
        return jsonify(flow.validate_tracking(os.path.join("videos", j["video"]), _name(j["video"]),
                                              int(j.get("frame", 150)), int(j.get("step", 1))))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/sfm", methods=["POST"])
def api_sfm():
    return jsonify(sfm.run())


if __name__ == "__main__":
    app.run(debug=True)
