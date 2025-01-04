import base64
import zlib


def decode_xmp(input_str: str) -> tuple[bytes, int]:
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

        return output, len(output)

    # Handle hex format
    if not all(c in "0123456789abcdef" for c in input_str):
        raise ValueError("Invalid hex data")

    output = bytes.fromhex(input_str)
    return output, len(output)
