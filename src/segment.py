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


def annotate_image(path: Path):
    # Get predictor and device using the new function
    predictor, device = get_predictor()

    # Load and prepare image
    image = cv2.imread(str(path))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize points
    points = []
    labels = []

    def mouse_callback(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONDOWN:  # Left click for positive points
            points.append([x, y])
            labels.append(1)
            cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)
            cv2.imshow("Image", display_image)
        elif event == cv2.EVENT_RBUTTONDOWN:  # Right click for negative points
            points.append([x, y])
            labels.append(0)
            cv2.circle(display_image, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Image", display_image)

    # Create window and set mouse callback
    cv2.namedWindow("Image")
    display_image = image.copy()
    cv2.imshow("Image", display_image)
    cv2.setMouseCallback("Image", mouse_callback)

    # Set image for predictor
    predictor.set_image(image_rgb)

    while True:
        key = cv2.waitKey(1) & 0xFF

        if key == ord("g"):  # Generate mask when 'g' is pressed
            if points:
                with torch.inference_mode(), torch.autocast(device.type, dtype=torch.float16):
                    masks, scores, _ = predictor.predict(
                        point_coords=np.array(points), point_labels=np.array(labels), multimask_output=False
                    )

                # Show mask overlay
                mask = masks[0].astype(bool)  # Convert to boolean array
                mask_overlay = display_image.copy()
                mask_overlay[mask] = mask_overlay[mask] * 0.5 + np.array([0, 0, 255]) * 0.5
                cv2.imshow("Mask", mask_overlay)

                # return masks

        elif key == ord("c"):  # Clear points when 'c' is pressed
            points.clear()
            labels.clear()
            display_image = image.copy()
            cv2.imshow("Image", display_image)

        elif key == ord("q"):  # Quit when 'q' is pressed
            break

    cv2.destroyAllWindows()
