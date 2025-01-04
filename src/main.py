import logging

import darktable

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Example mask string
    mask = "gz02eJwz8wuzu7WIy776joedrIy4/af9yXb6wX/szp7xsQVhRgYGhp9Oe+yqdn6161Rabjd5wju7rOSLdsmPUNXsffnSrk9WwX72og92PkuE7R8YPLa7ME3HHlnN0beT7by3uNqb8q202+/hab96Vq3ddXknFDUAo4M8/Q=="

    logging.info("Analyzing darktable path mask...\n")

    decoded, length = darktable.decode_xmp(mask)
    logging.info(decoded)
    logging.info(length)
