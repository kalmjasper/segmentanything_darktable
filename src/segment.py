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
        outline = extract_outline_from_mask(mask)

        if outline:  # Only draw if we have outline points
            outline_array = np.array(outline)
            cv2.polylines(display_image, [outline_array], isClosed=True, color=(0, 255, 0), thickness=2)

    # Redraw all points
    radius = max(5, min(display_image.shape[:2]) // 100)
    for (x_point, y_point), label in zip(points, labels, strict=True):
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        cv2.circle(display_image, (x_point, y_point), radius, color, -1)

    cv2.imshow(filename, display_image)


def extract_outline_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    """
    Extract the outline/contour points from a binary mask.

    Args:
        mask: Binary mask as numpy array

    Returns:
        List of (x,y) coordinate tuples forming the contour

    """
    # Convert mask to uint8 for contour detection
    mask_uint8 = mask.astype(np.uint8) * 255

    # Find contours
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get the largest contour
    if not contours:
        return []

    largest_contour = max(contours, key=cv2.contourArea)

    # Apply smoothing by approximating the polygon with fewer points
    epsilon = 0.0005 * cv2.arcLength(largest_contour, closed=True)  # 0.5% of perimeter
    smoothed_contour = cv2.approxPolyDP(largest_contour, epsilon, closed=True)

    # Convert contour points to list of tuples
    return [(int(point[0, 0]), int(point[0, 1])) for point in smoothed_contour]  # type: ignore


def normalize_outline(outline: list[tuple[int, int]], image: np.ndarray) -> list[tuple[float, float]]:
    height, width = image.shape[:2]
    return [(x / width, y / height) for x, y in outline]


def segment_outline(path: Path) -> list[tuple[float, float]]:
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

                    outline = extract_outline_from_mask(mask)
                    return normalize_outline(outline, image)

        finally:
            cv2.destroyAllWindows()
