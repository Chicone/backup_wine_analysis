import subprocess
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import yaml

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

scripts = {
    "bordeaux": "scripts/bordeaux/train_test_bordeaux.py",
    "press": "scripts/press_wines/train_test_press_wines.py",
    "pinot": "scripts/pinot_noir/train_test_pinot_noir.py",
}

from typing import List, Optional

class RunRequest(BaseModel):
    script_key: str
    classifier: Optional[str] = None
    feature_type: Optional[str] = None
    num_splits: Optional[int] = None
    normalize: Optional[bool] = None
    selected_datasets: Optional[List[str]] = None


from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import subprocess

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/run-script")
async def run_script(payload: dict):
    script_key = payload["script_key"]
    script = scripts.get(script_key)
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
    for key in ["selected_datasets", "classifier", "feature_type", "num_splits", "normalize", "sync_state",
                "class_by_year", "region", "show_confusion_matrix"]:
        if key in payload:
            config[key] = payload[key]

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
