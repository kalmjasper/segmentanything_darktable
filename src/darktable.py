import base64
import struct
import zlib
from dataclasses import dataclass
from enum import IntEnum


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
