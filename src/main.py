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
    segment.annotate_image(Path("test.jpeg"))
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    # else:
    #     device = torch.device("cpu")
    # print(f"using device: {device}")
    #
    # checkpoint = "./checkpoints/sam2.1_hiera_small.pt"
    # model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
    # predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device="mps"))
    #
    # with torch.inference_mode(), torch.autocast("mps", dtype=torch.float16):
    #
    #     image = Image.open("test.jpeg")
    #     image = np.array(image.convert("RGB"))
    #     predictor.set_image(image)
    #
    #     input_point = np.array([[500, 375]])
    #     input_label = np.array([1])
    #
    #     masks, scores, logits = predictor.predict(
    #         point_coords=input_point,
    #         point_labels=input_label,
    #         multimask_output=True,
    #     )
    #
    #     print(masks)
    #
