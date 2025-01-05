import argparse
import logging
from pathlib import Path

import darktable
import segment


def main():
    logging.basicConfig(level=logging.INFO)

    # masks = darktable.read_darktable_masks(
    #     Path("/Users/jasperinsinger/Documents/Piccas/Amsterdam 02-01-25/DSCF7193.RAF.xmp")
    # )
    #
    # for mask in masks:
    #     mask.points = [
    #         darktable.PathPoint(
    #             corner=(point.corner[0], point.corner[1]),
    #             ctrl1=point.ctrl1,
    #             ctrl2=point.ctrl2,
    #             border=point.border,
    #             state=darktable.PointState.NORMAL,
    #         )
    #         for point in mask.points
    #     ]
    #     point_bytes = darktable.encode_path_points(mask.points)
    #     point_str = darktable.encode_xmp(point_bytes)
    #
    #     for point in mask.points:
    #         print("corner")
    #         print(point.corner)
    #         print(point.ctrl1)
    #         print(point.ctrl2)
    #         print(point.border)
    #         print(point.state)
    #
    #     logging.info(point_str)

    parser = argparse.ArgumentParser(description="Generate Darktable masks from images using AI segmentation")
    parser.add_argument("--file", type=Path, help="Path to the image file")
    args = parser.parse_args()

    outline = segment.segment_outline(args.file)

    darktable_outline = [
        darktable.PathPoint(
            corner=(point[1], point[0]),
            ctrl1=(point[1], point[0]),
            ctrl2=(point[1], point[0]),
            border=(0.01, 0.01),
            state=darktable.PointState.NORMAL,
        )
        for point in outline
    ]

    logging.info(outline)

    points = darktable.encode_path_points(darktable_outline)
    point_str = darktable.encode_xmp(points)

    logging.info(point_str)


if __name__ == "__main__":
    main()
