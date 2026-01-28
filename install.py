try:
    from face_editor.io.util import load_classes_from_directory
except Exception:
    import os
    import sys

    sys.path.append(os.path.dirname(__file__))
    from face_editor.io.util import load_classes_from_directory

import traceback
import sys

import launch
from modules import shared

from face_editor.use_cases.installer import Installer

if sys.version_info >= (3, 12):
    print("Warning: Python versions beyond 3.11 are not officially supported by Face Editor yet.")

if not shared.opts:
    from modules import shared_init

    shared_init.initialize()

if shared.opts.data.get("face_editor_additional_components", None) is not None:
    for cls in load_classes_from_directory(Installer, True):
        try:
            cls().install()
        except Exception as e:
            print(traceback.format_exc())
            print(f"Face Editor: {e}")

launch.run_pip(
    "install lark",
    "requirements for Face Editor",
)
