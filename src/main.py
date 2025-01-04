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
        if mask.name == "HALLO":
            logging.info(mask)

            mask.points = [
                darktable.PathPoint(
                    corner=point.corner,
                    ctrl1=point.ctrl1,
                    ctrl2=point.ctrl2,
                    border=point.border,
                    state=darktable.PointState.NORMAL,
                )
                for point in mask.points
            ]

            logging.info(mask)

            logging.info("NEW POINTS")

    darktable.add_darktable_masks(
        Path("/Users/jasperinsinger/Documents/Piccas/Amsterdam 02-01-25/DSCF7089.RAF.xmp"), masks
    )
