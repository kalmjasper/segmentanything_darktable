import logging
from pathlib import Path

import darktable

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    masks = darktable.read_darktable_masks(
        Path("/Users/jasperinsinger/Documents/Piccas/Amsterdam 02-01-25/DSCF7089.RAF.xmp")
    )
    for mask in masks:
        logging.info(f"MASK {mask.name}")
        for point in mask.points:
            logging.info(point)
