from typing import List

from face_editor.use_cases.installer import Installer


class AnimeSegmentationInstaller(Installer):
    def name(self) -> str:
        return "AnimeSegmentation"

    def requirements(self) -> List[str]:
        reqs = ["huggingface_hub"]
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            reqs.append("onnxruntime")
        return reqs
