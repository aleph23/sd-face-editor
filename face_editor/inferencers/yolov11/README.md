# YOLOv11 components
Component implementation using [YOLOv11](https://github.com/ultralytics/ultralytics).

To use this, please enable 'yolov11' option under "Additional components" in the Face Editor section of the "Settings" tab.

## 1. Mask Generator
This component utilizes a [YOLOv11 Face Segmentation model](https://docs.ultralytics.com/tasks/segment/) (`yolo11n-face-seg.pt`) for precise face masking. Additionally, it uses a general object segmentation model (`yolo11n-seg.pt`) to detect and exclude occlusions (e.g., hands, accessories, phones) from the face mask, ensuring that only the visible face area is processed.

![Example](../../../images/inferencers/yolov11/mask.jpg)

#### Name
- YOLOv11

#### Implementation
- [Yolo11MaskGenerator](yolo11_mask_generator.py)

#### Recognized UI settings
- Use minimal area (for close faces)
- Mask size

#### Configuration Parameters (in JSON)
- `conf` (float, default: 0.4): Confidence threshold for face segmentation.
- `mask_conf` (float, default: 0.25): Confidence threshold for occlusion segmentation (hands, objects).

#### Usage in Workflows
- [yolov11.json](../../../workflows/examples/yolov11.json)
