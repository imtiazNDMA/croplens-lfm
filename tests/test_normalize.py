import io
import unittest

from PIL import Image

from normalize import SIDE, normalize_image


class NormalizeTests(unittest.TestCase):
    def test_every_shape_comes_out_the_same_size_and_format(self):
        for size in ((6000, 4000), (1040, 493), (256, 256), (300, 900)):
            result = Image.open(io.BytesIO(normalize_image(Image.new("RGB", size, "green"))))
            self.assertEqual((result.size, result.format), ((SIDE, SIDE), "JPEG"))

    def test_centre_crop_keeps_the_middle_and_drops_the_sides(self):
        image = Image.new("RGB", (300, 100), "red")
        image.paste(Image.new("RGB", (100, 100), "blue"), (100, 0))  # the middle square
        result = Image.open(io.BytesIO(normalize_image(image))).convert("RGB")
        r, g, b = result.getpixel((SIDE // 2, SIDE // 2))
        self.assertGreater(b, 200)
        self.assertLess(r, 60)
        r, g, b = result.getpixel((3, 3))  # a corner of the cropped middle is still blue, never the red sides
        self.assertGreater(b, 200)

    def test_metadata_is_dropped(self):
        image = Image.new("RGB", (64, 64), "green")
        exif = Image.Exif()
        exif[270] = "secret description"
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", exif=exif)
        buffer.seek(0)
        result = Image.open(io.BytesIO(normalize_image(Image.open(buffer))))
        self.assertEqual(len(result.getexif()), 0)


if __name__ == "__main__":
    unittest.main()
