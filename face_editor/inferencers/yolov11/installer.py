from launch import run_pip
from modules.paths_internal import models_path
import os
import requests

from face_editor.use_cases.installer import Installer as BaseInstaller


class Installer(BaseInstaller):
    def name(self) -> str:
        return "YOLOv11"

    def install(self) -> None:
        run_pip(
            "install ultralytics>=8.3.0 opencv-python-headless",
            "requirements for YOLOv11 inferencers of Face Editor",
        )

        # Define model paths
        yolo_path = os.path.join(models_path, "yolo")
        os.makedirs(yolo_path, exist_ok=True)

        models = [
            "yolo11n-face-seg.pt",
            "yolo11n-seg.pt"
        ]

        # Placeholder for Hugging Face repository
        # Replace 'USER/REPO' with the actual repository ID
        repo_id = "USER/REPO"

        for model in models:
            model_file = os.path.join(yolo_path, model)
            if not os.path.exists(model_file):
                print(f"Face Editor: Downloading {model}...")
                try:
                    url = f"https://huggingface.co/{repo_id}/resolve/main/{model}"
                    response = requests.get(url, stream=True)
                    if response.status_code == 200:
                        with open(model_file, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                f.write(chunk)
                        print(f"Face Editor: Successfully downloaded {model}")
                    else:
                        print(f"Face Editor: Failed to download {model} from {url} (Status: {response.status_code}). Please download it manually to {yolo_path}")
                except Exception as e:
                     print(f"Face Editor: Error downloading {model}: {e}")
