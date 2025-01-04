from pathlib import Path

import segment

# def main():
#     logging.basicConfig(level=logging.INFO)
#
#     masks = darktable.read_darktable_masks(
#         Path("/Users/jasperinsinger/Documents/Piccas/Amsterdam 02-01-25/DSCF7089.RAF.xmp")
#     )
#
#     for mask in masks:
#         mask.points = [
#             darktable.PathPoint(
#                 corner=(point.corner[0] - 0.4, point.corner[1]),
#                 ctrl1=point.ctrl1,
#                 ctrl2=point.ctrl2,
#                 border=point.border,
#                 state=darktable.PointState.NORMAL,
#             )
#             for point in mask.points
#         ]
#         point_bytes = darktable.encode_path_points(mask.points)
#         point_str = darktable.encode_xmp(point_bytes)
#
#         logging.info(point_str)
#

# Example usage
if __name__ == "__main__":
    segment.annotate_image(Path("/Users/jasperinsinger/Documents/Piccas/Amsterdam 02-01-25/DSCF7141.JPG"))
