import os

import launch
from face_editor.use_cases.installer import Installer


class OpenMMLabInstaller(Installer):
    def name(self) -> str:
        return "OpenMMLab"

    def install(self) -> None:
        launch.run_pip(
            "install --no-deps openxlab",
            "openxlab for occlusion masking",
        )
        launch.run_pip(
            'install "mmcv>=2.0.0" mmengine "mmsegmentation>=1.0.0" huggingface_hub mmdet',
            "requirements for openmmlab inferencers of Face Editor",
        )
