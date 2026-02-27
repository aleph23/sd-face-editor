
import cv2
import numpy as np
import os
from typing import List, Optional

from face_editor.entities.rect import Rect
from face_editor.use_cases.mask_generator import MaskGenerator
from modules.paths_internal import models_path


class Yolo11MaskGenerator(MaskGenerator):
    def name(self) -> str:
        return "YOLOv11"

    def generate_mask(self, image: np.ndarray, face_rect: Rect, **kwargs) -> np.ndarray:
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "Please enable 'YOLOv11' option under \"Additional components\" in the Face Editor section of the \"Settings\" tab and restart the WebUI."
            )

        conf = float(kwargs.get("conf", 0.4))
        mask_conf = float(kwargs.get("mask_conf", 0.25))

        yolo_path = os.path.join(models_path, "yolo")
        os.makedirs(yolo_path, exist_ok=True)

        face_model_path = os.path.join(yolo_path, "yolo11n-face-seg.pt")
        seg_model_path = os.path.join(yolo_path, "yolo11n-seg.pt")

        if not os.path.exists(face_model_path):
            print(f"Face Editor: Model not found at {face_model_path}. Please check installation or download manually.")
            return np.zeros(image.shape[:2], dtype=np.uint8)

        # 1. Get the "Positive" (Everything that IS face)
        try:
            face_model = YOLO(face_model_path)
            f_results = face_model(image, conf=conf, verbose=False)
        except Exception as e:
            print(f"Face Editor: Error loading/running face model: {e}")
            return np.zeros(image.shape[:2], dtype=np.uint8)

        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        if not f_results or not f_results[0].masks:
             return full_mask

        # Combine all detected face masks. Since we usually get a cropped face image,
        # we can assume the largest mask corresponds to the target face.
        best_mask = None
        max_area = 0

        for mask in f_results[0].masks.xy:
            if len(mask) > 0:
                poly = mask.astype(np.int32)
                area = cv2.contourArea(poly)
                if area > max_area:
                    max_area = area
                    best_mask = poly

        if best_mask is not None:
            cv2.fillPoly(full_mask, [best_mask], 255)
        else:
             return full_mask


        # 2. Get the "Negative" (Everything that BLOCKS face)
        if os.path.exists(seg_model_path):
            try:
                mask_model = YOLO(seg_model_path)
                m_results = mask_model(image, conf=mask_conf, verbose=False)

                if m_results and m_results[0].masks:
                    face_mask_area = np.sum(full_mask > 0)

                    for i, box in enumerate(m_results[0].boxes):
                        cls_id = int(box.cls)

                        # Get mask points
                        occ_poly_points = m_results[0].masks.xy[i]
                        if len(occ_poly_points) == 0:
                            continue
                        occ_poly = occ_poly_points.astype(np.int32)

                        should_subtract = True

                        # Special handling for 'person' (Class 0)
                        # The segmentation model often detects the face/body as a person.
                        # We want to subtract OTHER people's hands/parts, but not the face itself.
                        if cls_id == 0:
                             # Create a temporary mask for this object
                             obj_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                             cv2.fillPoly(obj_mask, [occ_poly], 255)

                             # Intersection with the face mask we just created
                             intersection = cv2.bitwise_and(full_mask, obj_mask)
                             inter_area = np.sum(intersection > 0)
                             obj_area = np.sum(obj_mask > 0)

                             if obj_area > 0:
                                 overlap_ratio = inter_area / obj_area

                                 # If the detected person object is mostly THE FACE ITSELF (high overlap),
                                 # do NOT subtract it.
                                 if overlap_ratio > 0.6: # Configurable threshold?
                                     should_subtract = False

                                 # Also consider relative size. If the object is HUGE (the whole body) vs the face mask,
                                 # we probably don't want to subtract it all unless it's blocking the face.
                                 # But overlap_ratio handles this: if body is huge, inter/body is small -> subtract?
                                 # Wait, if body is huge, subtracting it erases the face. BAD.

                                 # Let's flip it: Does the object CONTAIN the face?
                                 # If inter_area / face_mask_area > 0.8, then the object covers most of the face.
                                 # It is likely the person to whom the face belongs.
                                 if face_mask_area > 0:
                                     face_overlap = inter_area / face_mask_area
                                     if face_overlap > 0.8:
                                         should_subtract = False

                        if should_subtract:
                            # Dilate the occlusion mask slightly to ensure clean edges around the object
                            # We create a mask of the occlusion, dilate it, then subtract.
                            occ_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                            cv2.fillPoly(occ_mask, [occ_poly], 255)

                            kernel = np.ones((5, 5), np.uint8) # Slight dilation
                            dilated_occ = cv2.dilate(occ_mask, kernel, iterations=1)

                            # Subtract from full_mask
                            # full_mask = full_mask - dilated_occ (clipped at 0)
                            # Or simpler with bitwise_not
                            full_mask = cv2.bitwise_and(full_mask, cv2.bitwise_not(dilated_occ))

            except Exception as e:
                print(f"Face Editor: Error loading/running segmentation model: {e}")

        return full_mask
