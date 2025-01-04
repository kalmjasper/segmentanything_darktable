import logging
import os
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2  # type: ignore
from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore


def get_predictor() -> tuple[SAM2ImagePredictor, torch.device]:
    """
    Initialize and return a SAM2 predictor and the device it's using.

    Returns:
        tuple: (predictor, device) where predictor is a SAM2ImagePredictor and
        device is the torch.device being used

    """
    # Initialize device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if device.type == "mps":
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    logging.info(f"using device: {device}")

    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device=device))

    return predictor, device


def rescale_image(image: np.ndarray, max_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    scale = max_height / height
    new_width = int(width * scale)
    return cv2.resize(image, (new_width, max_height), interpolation=cv2.INTER_AREA)


def show_complete_image(
    image: np.ndarray, mask: np.ndarray, points: list[tuple[int, int]], labels: list[int], filename: str
):
    mask = mask.astype(bool)
    display_image = image.copy()

    # Only apply mask if there are masked pixels
    if mask.any():
        display_image[mask] = display_image[mask] * 0.5 + np.array([0, 0, 255]) * 0.5

    # Redraw all points
    radius = max(5, min(display_image.shape[:2]) // 100)
    for (x_point, y_point), label in zip(points, labels, strict=True):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(display_image, (x_point, y_point), radius, color, -1)

    cv2.imshow(filename, display_image)


def annotate_image(path: Path) -> np.ndarray:
    # Get predictor and device using the new function
    predictor, device = get_predictor()

    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16):
        # Load and prepare image
        image = cv2.imread(str(path))

        # Extract filename without extension using pathlib
        filename = path.stem

        # Rescale image to max 1080p while maintaining aspect ratio
        image = rescale_image(image, 1080)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Initialize points
        points = []
        labels = []

        # Create window and set mouse callback
        cv2.namedWindow(filename)

        show_complete_image(image, np.zeros_like(image), [], [], filename)

        mask = None

        def mouse_callback(event, x, y, _flags, _param):
            if event == cv2.EVENT_LBUTTONDOWN:  # Left click for positive points
                points.append([x, y])
                labels.append(1)
            elif event == cv2.EVENT_RBUTTONDOWN:  # Right click for negative points
                points.append([x, y])
                labels.append(0)

            if len(points) == 0:
                return

            masks, scores, _ = predictor.predict(
                point_coords=np.array(points), point_labels=np.array(labels), multimask_output=True
            )

            # Get mask with highest score
            best_mask_idx = np.argmax(scores)

            nonlocal mask  # Save the mask to return later
            mask = masks[best_mask_idx].astype(bool)

            show_complete_image(image, mask, points, labels, filename)

        cv2.setMouseCallback(filename, mouse_callback)

        # Set image for predictor
        predictor.set_image(image_rgb)

        try:
            while True:
                key = cv2.waitKey(1) & 0xFF

                if key == ord("c"):  # Clear points when 'c' is pressed
                    points.clear()
                    labels.clear()
                    display_image = image.copy()
                    cv2.imshow(filename, display_image)

                elif key == ord("q"):  # Quit when 'q' is pressed
                    if mask is None:
                        raise ValueError("No mask found")
                    return mask

        finally:
            cv2.destroyAllWindows()
