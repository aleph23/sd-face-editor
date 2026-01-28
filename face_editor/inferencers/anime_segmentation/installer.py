from typing import List
import launch
import importlib.util

from face_editor.use_cases.installer import Installer


class AnimeSegmentationInstaller(Installer):
    def name(self) -> str:
        return "AnimeSegmentation"

    def requirements(self) -> List[str]:
        reqs = ["huggingface_hub"]
        if importlib.util.find_spec("onnxruntime") is None:
            reqs.append("onnxruntime")
        return reqs
