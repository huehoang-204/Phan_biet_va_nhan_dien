#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import uuid
from typing import Dict, List
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, abort

# Paths
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
PUBLIC_OUT_DIR = os.path.join(PROJECT_ROOT, "labeled_outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Import pipeline
import sys
sys.path.insert(0, PROJECT_ROOT)
from chayngaydi import run_pipeline  # noqa: E402

app = Flask(__name__, static_folder="static", template_folder="templates")
# Không cho trình duyệt cache trang kết quả (tránh F5 hiện lại trang cũ)
@app.after_request
def add_no_cache_headers(resp):
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp

app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024  # 1GB


@app.route("/")
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return redirect(url_for("index"))
    f = request.files["audio"]
    if not f or f.filename == "":
        return redirect(url_for("index"))

    # --- chỉ còn 1 lựa chọn: export_mode ---
    export_mode = (request.form.get("export_mode") or "concat").strip().lower()
    if export_mode not in {"concat", "full", "segments"}:
        export_mode = "concat"

    # Không dùng num_speakers nữa (để None => NeMo tự suy)
    num_speakers = None

    # 'concat' của bạn là "ghép liền" ⇒ không có gap
    gap = 0.0

    # save upload
    job_id = uuid.uuid4().hex[:12]
    ext = os.path.splitext(f.filename)[1].lower()
    up_name = f"{job_id}{ext or '.wav'}"
    up_path = os.path.join(UPLOAD_DIR, up_name)
    f.save(up_path)

    # run pipeline
    result = run_pipeline(
        input_wav=up_path,
        num_speakers=num_speakers,
        mode=export_mode,
        concat_gap_sec=gap,
        normalize=True,
    )

    # build URLs (không còn with_silence)
    final_dir = result["final_dir"]
    files: List[str] = result["files"]
    rel_files = []
    for p in files:
        rel = os.path.relpath(p, PUBLIC_OUT_DIR)
        base = os.path.splitext(os.path.basename(p))[0]
        label = base.split("_")[-1] if "_" in base else base
        rel_files.append({"label": label, "url": url_for("serve_output", path=rel.replace("\\", "/"))})

    input_rel = os.path.relpath(up_path, UPLOAD_DIR)
    input_url = url_for("serve_upload", path=input_rel.replace("\\", "/"))
    out_rel = os.path.relpath(final_dir, PUBLIC_OUT_DIR)
    diarization = result.get("diarization", [])

    # Input waveform
    input_waveform_rel = result.get("input_waveform_url")
    input_waveform_url = (
        url_for("serve_output", path=input_waveform_rel.replace("\\", "/")) + f"?v={uuid.uuid4().hex[:6]}"
        if input_waveform_rel else None
    )

    # Output waveform (ẩn khi concat như bạn đang làm)
    output_waveform_rel = result.get("output_waveform_url")
    if export_mode == "concat":
        output_waveform_url = None
    else:
        output_waveform_url = (
            url_for("serve_output", path=output_waveform_rel.replace("\\", "/")) + f"?v={uuid.uuid4().hex[:6]}"
            if output_waveform_rel else None
        )

    #output_waveform_url = url_for("serve_output", path=output_waveform_rel.replace("\\", "/")) if output_waveform_rel else None

    #speaker_waves = result.get("speaker_waveforms", {})
    #for sp in rel_files:
    #    lbl = sp["label"]
    #    sp["waveform_url"] = url_for("serve_output", path=speaker_waves[lbl].replace("\\", "/")) if lbl in speaker_waves else None
    # Speaker waveforms
    speaker_waves = result.get("speaker_waveforms", {})
    for sp in rel_files:
        label = sp["label"]
        if label in speaker_waves:
            # thêm cache-busting query để trình duyệt không giữ ảnh cũ
            sp["waveform_url"] = url_for("serve_output", path=speaker_waves[label].replace("\\", "/")) + f"?v={uuid.uuid4().hex[:6]}"
        else:
            sp["waveform_url"] = None

    return render_template(
        "index.html",
        job_id=job_id,
        input_audio_url=input_url,
        output_dir_url=url_for("serve_output_dir", path=out_rel.replace("\\", "/")),
        diarization=diarization,
        output_audio_url=input_url,
        input_waveform_url=input_waveform_url,
        output_waveform_url=output_waveform_url,
        speakers=rel_files,
        export_mode=export_mode,   
    )


@app.route("/uploads/<path:path>")
def serve_upload(path: str):
    return send_from_directory(UPLOAD_DIR, path)


@app.route("/outputs/<path:path>")
def serve_output(path: str):
    return send_from_directory(PUBLIC_OUT_DIR, path)


@app.route("/outputs_dir/<path:path>")
def serve_output_dir(path: str):
    # Browsing helper + harden path traversal
    requested = os.path.normpath(os.path.join(PUBLIC_OUT_DIR, path))
    public_root = os.path.realpath(PUBLIC_OUT_DIR)
    requested_real = os.path.realpath(requested)

    # chặn truy cập ra ngoài thư mục public
    if requested_real != public_root and not requested_real.startswith(public_root + os.sep):
        abort(403)

    if not os.path.isdir(requested_real):
        return jsonify({"error": "NotFound"}), 404

    files = sorted(os.listdir(requested_real))
    return jsonify({"files": files})


def create_app():
    return app


if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "7860"))
    app.run(host=host, port=port, debug=True)
