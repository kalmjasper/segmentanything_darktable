import base64
import logging
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum, IntFlag
from pathlib import Path

from defusedxml import ElementTree as DefusedET  # type: ignore


class MaskType(IntFlag):
    """Darktable mask types."""

    NONE = 0
    CIRCLE = 1 << 0
    PATH = 1 << 1
    GROUP = 1 << 2
    CLONE = 1 << 3
    GRADIENT = 1 << 4
    ELLIPSE = 1 << 5
    BRUSH = 1 << 6
    NON_CLONE = 1 << 7


class PointState(IntEnum):
    """State of a path point."""

    NORMAL = 1
    USER = 2


@dataclass
class PathPoint:
    """A point in a path mask."""

    corner: tuple[float, float]  # x,y coordinates of the point
    ctrl1: tuple[float, float]  # x,y coordinates of the first control point
    ctrl2: tuple[float, float]  # x,y coordinates of the second control point
    border: tuple[float, float]  # x,y coordinates of the border point
    state: PointState  # state of the point


@dataclass
class DarktablePathMask:
    """A darktable path mask."""

    points: list[PathPoint]  # List of path points
    name: str
    version: int


def read_darktable_masks(path: Path) -> list[DarktablePathMask]:
    """
    Read darktable masks from an XMP file.

    Args:
        path: Path to the XMP file

    Returns:
        List of DarktablePathMask objects, containing only Path type masks

    """
    tree = DefusedET.parse(path)
    root = tree.getroot()

    # Find the masks_history sequence
    ns = {
        "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        "darktable": "http://darktable.sf.net/",
    }
    masks_seq = root.find(".//darktable:masks_history/rdf:Seq", ns)

    if masks_seq is None:
        return []

    masks = []
    for mask in masks_seq.findall("rdf:li", ns):
        # Check if this is a path mask (type 2)
        mask_type = int(mask.get(f"{{{ns['darktable']}}}mask_type", "0"))
        if mask_type != MaskType.PATH:
            logging.warning(f"Skipping mask type {mask_type}")
            continue

        mask_points = mask.get(f"{{{ns['darktable']}}}mask_points", "")
        if not mask_points:
            continue

        # Decode and parse the points
        decoded = decode_xmp(mask_points)
        points = parse_path_points(decoded)

        masks.append(
            DarktablePathMask(
                points=points,
                # mask_id=int(mask.get(f"{{{ns['darktable']}}}mask_id", "0")),
                name=mask.get(f"{{{ns['darktable']}}}mask_name", ""),
                version=int(mask.get(f"{{{ns['darktable']}}}mask_version", "0")),
            )
        )

    return masks


def decode_xmp(input_str: str) -> bytes:
    """
    Decode XMP data from darktable format.

    Args:
        input_str: Input string in either gz-compressed base64 or hex format

    Returns:
        Tuple of (decoded bytes, length)

    """
    # Check if data is compressed (starts with gz)
    if input_str.startswith("gz"):
        # Get compression factor from next 2 chars
        factor = 10 * (ord(input_str[2]) - ord("0")) + (ord(input_str[3]) - ord("0"))

        # Decode base64 data after gz## prefix
        compressed = base64.b64decode(input_str[4:])

        # Try decompressing with increasing buffer sizes
        output = None
        buf_len = int(factor * len(compressed))

        max_iters = 1000000

        while True:
            try:
                output = zlib.decompress(compressed, bufsize=buf_len)
                break
            except zlib.error:
                if buf_len > max_iters:  # Prevent infinite loop
                    raise
                buf_len *= 2

        return output

    # Handle hex format
    if not all(c in "0123456789abcdef" for c in input_str):
        raise ValueError("Invalid hex data")

    return bytes.fromhex(input_str)


def encode_xmp(input_bytes: bytes) -> str:
    """
    Encode bytes into XMP data in darktable's gz-compressed base64 format.

    Args:
        input_bytes: Raw bytes to encode

    Returns:
        String in gz-compressed base64 format with gz## prefix

    """
    # Compress the input bytes
    compressed = zlib.compress(input_bytes)

    # Calculate compression factor (rounded to nearest int)
    factor = max(1, min(99, round(len(input_bytes) / len(compressed))))

    # Convert to base64
    b64_data = base64.b64encode(compressed).decode("ascii")

    # Add gz prefix with compression factor
    return f"gz{factor:02d}{b64_data}"


def parse_path_points(points_raw: bytes) -> list[PathPoint]:
    """
    Parse raw bytes into a list of path points.

    Args:
        points_raw: Raw bytes containing path point data

    Returns:
        List of PathPoint objects

    """
    # Size of one point struct (8 floats + 1 int)
    point_size = 8 * 4 + 4  # 36 bytes

    if len(points_raw) % point_size != 0:
        raise ValueError(f"Input bytes length {len(points_raw)} is not a multiple of point size {point_size}")

    num_points = len(points_raw) // point_size
    points: list[PathPoint] = []

    for i in range(num_points):
        offset = i * point_size
        # Unpack 8 floats and 1 int
        values = struct.unpack("8fi", points_raw[offset : offset + point_size])

        points.append(
            PathPoint(
                corner=(values[0], values[1]),
                ctrl1=(values[2], values[3]),
                ctrl2=(values[4], values[5]),
                border=(values[6], values[7]),
                state=PointState(values[8]),
            )
        )

    return points


def encode_path_points(path_points: list[PathPoint]) -> bytes:
    """
    Encode a list of path points into raw bytes.

    Args:
        path_points: List of PathPoint objects to encode

    Returns:
        Raw bytes containing encoded path point data

    """
    points_raw = bytearray()

    for point in path_points:
        # Pack 8 floats and 1 int into bytes
        points_raw.extend(
            struct.pack(
                "8fi",
                point.corner[0],
                point.corner[1],
                point.ctrl1[0],
                point.ctrl1[1],
                point.ctrl2[0],
                point.ctrl2[1],
                point.border[0],
                point.border[1],
                point.state.value,
            )
        )

    return bytes(points_raw)
