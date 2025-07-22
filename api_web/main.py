import subprocess
import os
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml
import asyncio
from fastapi.responses import StreamingResponse
import subprocess
import datetime

app = FastAPI()
log_queue = asyncio.Queue()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scripts = {
    "bordeaux": "scripts/bordeaux/train_test_bordeaux.py",
    "press": "scripts/press_wines/train_test_press_wines.py",
    "pinot": "scripts/pinot_noir/train_test_pinot_noir.py",
    "champagne_predict_labels": "scripts/champagne/train_test_labels.py",
    "champagne_predict_age": "scripts/champagne/train_test_age.py",
    "champagne_global_model": "scripts/champagne/ridge_model_global.py",
    "champagne_per_taster_model": "scripts/champagne/ridge_model_per_taster.py",

    "bordeaux_projection": "scripts/bordeaux/train_test_bordeaux.py",
    # "bordeaux_projection": "scripts/bordeaux/projection_bordeaux.py",
    "pinot_projection": "scripts/pinot_noir/train_test_pinot_noir.py",
    # "pinot_projection": "scripts/pinot_noir/old_projection_pinot_noir.py",
    "press_projection": "scripts/press_wines/train_test_press_wines.py",
    # "press_projection": "scripts/press_wines/projection_press_wines.py",
    "champagne_global_model_projection":  "scripts/champagne/ridge_model_global.py",

}

from typing import List, Optional

class RunRequest(BaseModel):
    script_key: str
    classifier: Optional[str] = None
    feature_type: Optional[str] = None
    num_repeats: Optional[int] = None
    normalize: Optional[bool] = None
    selected_datasets: Optional[List[str]] = None


async def logMessage(message: str):
    timestamp = datetime.datetime.now().strftime("[%I:%M:%S %p]")
    full_message = f"{timestamp} {message}"
    await log_queue.put(full_message)
    print(full_message)  # Optional: still prints to backend console

@app.get("/logs")
async def stream_logs():
    async def event_generator():
        last_pos = 0
        try:
            with open("log_buffer.txt", "r") as f:
                full = f.read()
                if full:
                    for line in full.splitlines():
                        yield f"data: {line}\n\n"
                last_pos = f.tell()
        except FileNotFoundError:
            pass

        while True:
            await asyncio.sleep(1)
            try:
                with open("log_buffer.txt", "r") as f:
                    f.seek(last_pos)
                    new = f.read()
                    last_pos = f.tell()
                    if new:
                        yield f"data: {new}\n\n"
            except FileNotFoundError:
                pass

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/run-script")
async def run_script(payload: dict):
    script_key = payload["script_key"]
    if script_key == "champagne_predict_labels":
        key = script_key
    else:
        key = f"{script_key}_projection" if payload.get("plot_projection") else script_key
    script = scripts.get(key)
    if not script:
        return StreamingResponse(iter(["Invalid script selected.\n"]), media_type="text/plain")

    # Full script path
    full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", script))
    config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))

    # Load and modify config.yaml
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        return StreamingResponse(iter([f"Failed to read config.yaml: {e}\n"]), media_type="text/plain")

    # Apply overrides from frontend
    optional_keys = [
        ("selected_datasets", []),
        ("classifier", "RGC"),
        ("regressor", "ridge"),
        ("feature_type", "TIC"),
        ("num_repeats", 5),
        ("chrom_cap", 35000),
        ("normalize", False),
        ("sync_state", False),
        ("class_by_year", False),
        ("show_confusion_matrix", False),
        ("plot_projection", False),
        ("color_by_country", False),
        ("projection_source", "scores"),
        ("projection_dim", 2),
        ("projection_method", "UMAP"),
        ("n_neighbors", 15),
        ("random_state", 42),
        ("region", "region"),
        ("label_targets", "taster"),
        ("show_sample_names", "showSampleNames"),
        ("show_pred_plot", "show_pred_plot"),
        ("show_age_histogram", "show_age_histogram"),
        ("show_chromatograms", "show_chromatograms"),
        ("rt_range", "rt_range"),
        ("show_predicted_profiles", False),
        ("taster_scaling", False),
        ("shuffle_labels", False),
        ("test_average_scores", False),
        ("taster_vs_mean", False),
        ("plot_all_tests", "plot_all_tests"),
        ("group_wines", False),
        ("cv_type", "LOOPC"),
        ("invert_x", "invert_x"),
        ("invert_y", "invert_y"),
        ("global_focus_heatmap", "global_focus_heatmap"),
        ("taster_focus_heatmap", "taster_focus_heatmap"),
        ("plot_r2", "plot_r2"),
        ("plot_shap", "plot_shap"),
        ("reduce_dims", "reduce_dims"),
        ("reduction_method", "reduction_method"),
        ("reduction_dims", "reduction_dims"),
        ("remove_avg_scores", "remove_avg_scores"),
        ("constant_ohe", "constant_ohe"),
        ("do_classification", False),
        ("selected_attribute", "fruity"),
        ("sample_display_mode", "names"),
        ("color_by_winery", False),
        ("color_by_origin", False),
        ("exclude_us", False),
        ("density_plot", False),
    ]
    for key, default in optional_keys:
        if key in payload:
            config[key] = payload[key]
        elif key not in config:
            config[key] = default
    if script_key == "champagne_predict_labels":
        for unused in ["sync_state", "class_by_year", "feature_type"]:
            config.pop(unused, None)

    # if "region" in config:
    #     if not config["region"] or config["region"].strip() == "":
    #         del config["region"]

    # Save updated config.yaml
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f)
    except Exception as e:
        return StreamingResponse(iter([f"Failed to write config.yaml: {e}\n"]), media_type="text/plain")

    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    def run_and_stream():
        try:
            process = subprocess.Popen(
                ["python", full_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env=env,
                bufsize=1
            )
            for line in process.stdout:
                yield line
        except Exception as e:
            yield f"Error running script: {str(e)}\n"

    return StreamingResponse(run_and_stream(), media_type="text/plain")


# @app.post("/run-script")
# async def run_script(payload: dict):
#     script_key = payload["script_key"]
#     def run_and_stream():
#         process = subprocess.Popen(
#             ["python", f"{script_key}.py"],
#             stdout=subprocess.PIPE,
#             stderr=subprocess.STDOUT,
#             text=True,
#             bufsize=1
#         )
#         for line in process.stdout:
#             yield line
#
#     return StreamingResponse(run_and_stream(), media_type="text/plain")

# @app.post("/run-script")
# def run_script(req: RunRequest):
#     script = scripts.get(req.script_key)
#     if not script:
#         return {"status": "error", "message": "Invalid script selected."}
#
#     # 🔹 Full absolute path to the script
#     full_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", script))
#
#     # 🔹 Load the existing config.yaml
#     config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
#     try:
#         with open(config_path, "r") as f:
#             config = yaml.safe_load(f)
#     except Exception as e:
#         return {"status": "error", "message": f"Failed to read config.yaml: {e}"}
#
#     # 🔹 Apply overrides from the frontend
#     if req.selected_datasets:
#         config["selected_datasets"] = req.selected_datasets
#     if req.classifier:
#         config["classifier"] = req.classifier
#     if req.feature_type:
#         config["feature_type"] = req.feature_type
#     if req.num_splits is not None:
#         config["num_splits"] = req.num_splits
#     if req.normalize is not None:
#         config["normalize"] = req.normalize
#
#     # 🔹 Save updated config.yaml
#     try:
#         with open(config_path, "w") as f:
#             yaml.dump(config, f)
#     except Exception as e:
#         return {"status": "error", "message": f"Failed to write config.yaml: {e}"}
#
#     # 🔹 Set PYTHONPATH and run the script
#     env = os.environ.copy()
#     env["PYTHONPATH"] = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
#
#     try:
#         output = subprocess.check_output(
#             ["python", full_path],
#             stderr=subprocess.STDOUT,
#             text=True,
#             env=env
#         )
#         return {"status": "success", "output": output}
#     except subprocess.CalledProcessError as e:
#         return {"status": "error", "output": e.output}
