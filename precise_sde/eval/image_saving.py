import os
import struct
import zlib


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _png_chunk(chunk_type, payload):
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def save_rgb_png(image_rows, output_path):
    if not image_rows or not image_rows[0]:
        raise ValueError("image_rows must be a non-empty HxWx3 image")

    height = len(image_rows)
    width = len(image_rows[0])
    encoded_rows = []

    for row in image_rows:
        if len(row) != width:
            raise ValueError("all rows must share the same width")
        flat_row = []
        for pixel in row:
            if len(pixel) != 3:
                raise ValueError("each pixel must contain exactly 3 channels")
            for channel in pixel:
                channel = int(channel)
                if channel < 0 or channel > 255:
                    raise ValueError("pixel values must be between 0 and 255")
                flat_row.append(channel)
        encoded_rows.append(b"\x00" + bytes(flat_row))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    idat = zlib.compress(b"".join(encoded_rows))

    with open(output_path, "wb") as handle:
        handle.write(PNG_SIGNATURE)
        handle.write(_png_chunk(b"IHDR", ihdr))
        handle.write(_png_chunk(b"IDAT", idat))
        handle.write(_png_chunk(b"IEND", b""))


def save_generated_images_as_pngs(indexed_images, output_dir):
    image_dir = os.path.join(output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    image_paths = {}
    for index, image_rows in indexed_images:
        image_path = os.path.join(image_dir, f"{int(index):06d}.png")
        save_rgb_png(image_rows, image_path)
        image_paths[int(index)] = image_path
    return image_paths
