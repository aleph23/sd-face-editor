from typing import List
import launch
import importlib.util

from face_editor.use_cases.installer import Installer


class InsightFaceInstaller(Installer):
    def name(self) -> str:
        return "InsightFace"

    def requirements(self) -> List[str]:
        reqs = ['"insightface>=0.7.3"']
        if importlib.util.find_spec("onnxruntime") is None:
            reqs.append("onnxruntime")
        return reqs

    def install(self) -> None:
        try:
            from face_editor.inferencers.insightface.detector import InsightFaceDetector

            InsightFaceDetector()
        except Exception:
            super().install()
        return None
