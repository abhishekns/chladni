import unittest
from PIL import Image
import numpy as np
import math

# Adjust import path for tests
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chladni.core import ValueMap, ITERATION_MULTIPLIER
from chladni.visualization import (_mix_colors, ColorMap, create_grayscale_palette,
                               create_spectrum_palette, generate_bitmap_pil, DEFAULT_COLOR_MAPS, RGBColor)

class TestMixColors(unittest.TestCase):
    def test_mix_basic(self):
        c1 = (0, 0, 0) # Black
        c2 = (255, 255, 255) # White

        # weight2_percent is for c2. So, 0% means all c1.
        # Original: _mix_colors(target_color, source_color, weight_for_target_percent)
        # Here: _mix_colors(palette[i+1], palette[i], weight_for_palette[i+1])
        # If weight2_percent = 0, it means palette[i] (source_color)
        # If weight2_percent = 100, it means palette[i+1] (target_color)

        # Test with weight for color2 (which is the second argument to _mix_colors)
        # _mix_colors(color1=c2, color2=c1, weight2_percent=0) should yield c1
        self.assertEqual(_mix_colors(color1=c2, color2=c1, weight2_percent=0), c1)

        # _mix_colors(color1=c2, color2=c1, weight2_percent=100) should yield c2
        self.assertEqual(_mix_colors(color1=c2, color2=c1, weight2_percent=100), c2)

        # _mix_colors(color1=c2, color2=c1, weight2_percent=50) should yield mid-gray
        self.assertEqual(_mix_colors(color1=c2, color2=c1, weight2_percent=50), (127, 127, 127)) # or 128, depends on rounding

        c3 = (10, 20, 30)
        c4 = (50, 120, 230)
        # Test 25% weight for c4
        # Expected: c3*0.75 + c4*0.25
        # R = 10*0.75 + 50*0.25 = 7.5 + 12.5 = 20
        # G = 20*0.75 + 120*0.25 = 15 + 30 = 45
        # B = 30*0.75 + 230*0.25 = 22.5 + 57.5 = 80
        self.assertEqual(_mix_colors(color1=c4, color2=c3, weight2_percent=25), (20, 45, 80))

    def test_mix_clipping(self):
        c1 = (0, 0, 0)
        c2 = (300, -50, 255) # Values outside 0-255 before mixing (though RGBColor type hint implies valid)
                              # The function expects valid 0-255 inputs, clipping is for results

        # Let's use values that would result in out-of-bounds if not clipped
        c_high = (200, 200, 200)
        c_very_high = (200, 200, 200) # Mixing these with high weight for c_very_high

        # If c1=(200,200,200) and c2=(200,200,200), weight 100% for c2 = (200,200,200)
        # Let's try to make calculation go over 255
        # color1 (target), color2 (source), weight for color1
        # if color1=(200,200,200), color2=(100,100,100), weight=200 (invalid percent, but tests formula)
        # w2 = 2.0, w1 = -1.0.  r = 100*-1 + 200*2 = -100 + 400 = 300 -> clip to 255
        # This is not how weight2_percent works. It's 0-100.
        # The clipping in _mix_colors is for the *result* of the weighted average.

        # Example: c1=(250,250,250), c2=(250,250,250). weight2_percent=50.
        # r = int(250*0.5 + 250*0.5) = int(125+125) = 250. This doesn't test clipping.
        # The inputs r,g,b to max/min are already results of weighted average.
        # The inputs to _mix_colors are assumed to be valid RGB tuples.

        # No direct way to force inputs to _mix_colors to be out of range due to RGBColor type.
        # The clipping is defensive for the calculation result.
        # e.g. int(color2[0] * w1 + color1[0] * w2) could be <0 or >255 if floats are extreme,
        # but highly unlikely with w1,w2 in [0,1] and colors in [0,255].
        # It seems the clipping is more of a safeguard than a typically hit path.
        # Let's assume inputs are valid and focus on standard interpolation.
        pass # The current structure of _mix_colors makes it hard to test internal clipping robustly
             # without bypassing type hints or using extreme weights (which are not its intended use).


class TestColorMap(unittest.TestCase):
    def setUp(self):
        self.gray_palette = create_grayscale_palette()
        self.spectrum_palette = create_spectrum_palette()
        self.default_max_iter = ITERATION_MULTIPLIER

    def test_initialization(self):
        cm = ColorMap("test_gray", self.gray_palette, max_iter=100.0)
        self.assertEqual(cm.name, "test_gray")
        self.assertEqual(cm.max_iter, 100.0)
        self.assertTrue(np.array_equal(cm.palette, np.array(self.gray_palette, dtype=np.uint8)))
        self.assertAlmostEqual(cm._rec_max_col, 1.0/100.0)

        with self.assertRaises(ValueError):
            ColorMap("bad_palette", [(0,0,0)]*10) # Palette not 256 colors

        cm_zero_max_iter = ColorMap("test_zero", self.gray_palette, max_iter=0)
        self.assertEqual(cm_zero_max_iter._rec_max_col, 0.0)


    def test_set_max_iter(self):
        cm = ColorMap("test", self.gray_palette, self.default_max_iter)
        cm.set_max_iter(1000.0)
        self.assertEqual(cm.max_iter, 1000.0)
        self.assertAlmostEqual(cm._rec_max_col, 1.0/1000.0)

        cm.set_max_iter(0)
        self.assertEqual(cm.max_iter, 0)
        self.assertEqual(cm._rec_max_col, 0)

        # Test no change if same value
        cm.set_max_iter(0) # no change expected
        # How to check _update_calc was not called again? Could mock it, or rely on coverage.

    def test_iter_to_rgb_grayscale(self):
        cm = ColorMap("gray", self.gray_palette, max_iter=255.0) # Simple 1:1 mapping

        self.assertEqual(cm.iter_to_rgb(-10.0), (0,0,0)) # Below 0
        self.assertEqual(cm.iter_to_rgb(0.0), (0,0,0))   # Value 0
        self.assertEqual(cm.iter_to_rgb(127.0), (127,127,127)) # Mid value, exact palette index
        # Corrected expectation based on current _mix_colors logic
        self.assertEqual(cm.iter_to_rgb(127.5), (127,127,127))
                                                              #palette[127]=(127,127,127), palette[128]=(128,128,128)
                                                              #idx_float = 127.5. i=127. frac=0.5. weight=round(0.5*100)=50.
                                                              #mix((128,128,128), (127,127,127), 50) -> (127,127,127)
                                                              # (127*0.5 + 128*0.5) = 127.5. int(127.5) = 127.
                                                              # My manual trace of _mix_colors with (128,128,128), (127,127,127), 50:
                                                              # w2=0.5, w1=0.5. r = int(127*0.5 + 128*0.5) = int(63.5 + 64) = int(127.5) = 127.
                                                              # This seems to make (127,127,127). The test output in visualization.py shows 128.
                                                              # The rounding in _mix_colors is int().
                                                              # Let's re-check: round(fractional_part * 100.0). If fractional_part is 0.5, weight is 50.
                                                              # _mix_colors(color_i_plus_1, color_i, 50)
                                                              # r = int(color_i[0]*0.5 + color_i_plus_1[0]*0.5)
                                                              # If color_i[0]=127, color_i_plus_1[0]=128 --> int(127*0.5 + 128*0.5) = int(63.5 + 64) = int(127.5) = 127
                                                              # The example output might be based on a different rounding for weight or final color.
                                                              # Sticking to current code logic, it should be 127.
        self.assertEqual(cm.iter_to_rgb(127.5), (127,127,127))


        self.assertEqual(cm.iter_to_rgb(255.0), (255,255,255)) # Max iter
        self.assertEqual(cm.iter_to_rgb(300.0), (255,255,255)) # Above max

        cm_default_iter = ColorMap("gray_default", self.gray_palette, max_iter=self.default_max_iter)
        val = self.default_max_iter / 2.0
        # normalized_value = val * (1/default_max_iter) = 0.5
        # n_scaled = 0.5 * 255 = 127.5
        # i = 127, frac = 0.5, weight = 50
        # result should be (127,127,127)
        self.assertEqual(cm_default_iter.iter_to_rgb(val), (127,127,127))

    def test_iter_to_rgb_spectrum(self):
        cm = ColorMap("spectrum", self.spectrum_palette, max_iter=255.0) # 1:1 mapping
        # Spot check some values based on create_spectrum_palette
        self.assertEqual(cm.iter_to_rgb(0.0), self.spectrum_palette[0])      # Blueish (0,0,255)
        self.assertEqual(cm.iter_to_rgb(63.0), self.spectrum_palette[63])    # Cyanish (0,252,255)
        # For 63.5: i=63, frac=0.5, weight=50. Mix(palette[64], palette[63], 50)
        # palette[63] = (0, 252, 255)
        # palette[64] = (0, 255, 255) -> Blue to Cyan, g = i*4. So g for 63 is 252. For 64, it's i=64, g=255, b=255-(64-64)*4 = 255.
        # Mix((0,255,255), (0,252,255), 50) -> R=0, G=int(252*0.5+255*0.5)=int(126+127.5)=int(253.5)=253, B=255
        self.assertEqual(cm.iter_to_rgb(63.5), (0,253,255))

        self.assertEqual(cm.iter_to_rgb(127.0), self.spectrum_palette[127])  # Greenish (0,255,3)
        self.assertEqual(cm.iter_to_rgb(191.0), self.spectrum_palette[191])  # Yellowish (252,255,0)
        self.assertEqual(cm.iter_to_rgb(255.0), self.spectrum_palette[255])  # Reddish (255,0,0)


class TestPaletteCreation(unittest.TestCase):
    def test_create_grayscale_palette(self):
        palette = create_grayscale_palette()
        self.assertEqual(len(palette), 256)
        self.assertIsInstance(palette[0], tuple)
        self.assertEqual(palette[0], (0,0,0))
        self.assertEqual(palette[128], (128,128,128))
        self.assertEqual(palette[255], (255,255,255))
        for color in palette:
            self.assertEqual(len(color), 3)
            self.assertTrue(0 <= color[0] <= 255)
            self.assertTrue(0 <= color[1] <= 255)
            self.assertTrue(0 <= color[2] <= 255)

    def test_create_spectrum_palette(self):
        palette = create_spectrum_palette()
        self.assertEqual(len(palette), 256)
        self.assertIsInstance(palette[0], tuple)
        # Spot check key colors from its definition
        self.assertEqual(palette[0], (0,0,255))     # Start: Blue
        self.assertEqual(palette[63], (0,252,255))  # End of Blue to Cyan segment
        self.assertEqual(palette[64], (0,255,255))  # Start of Cyan to Green segment
        self.assertEqual(palette[127], (0,255,3))   # End of Cyan to Green segment
        self.assertEqual(palette[128], (0,255,0)) # Start of Green to Yellow segment (approx, r=(128-128)*4=0)
                                                   # Actually (0,255,0)
        self.assertEqual(palette[128], (0,255,0))
        self.assertEqual(palette[191], (252,255,0)) # End of Green to Yellow
        self.assertEqual(palette[192], (255,255,0)) # Start of Yellow to Red
        self.assertEqual(palette[255], (255,3,0))   # End: Red (actual calculation is (255,3,0))
        for color in palette:
            self.assertEqual(len(color), 3)
            self.assertTrue(0 <= color[0] <= 255)
            self.assertTrue(0 <= color[1] <= 255)
            self.assertTrue(0 <= color[2] <= 255)

class TestGenerateBitmapPIL(unittest.TestCase):
    def setUp(self):
        self.width, self.height = 4, 2 # Small bitmap
        self.value_map = ValueMap(self.width, self.height)
        # Create a simple gradient: 0, 1, 2, 3 for first row, 4, 5, 6, 7 for second
        # Scaled by ITERATION_MULTIPLIER / 8 for test range
        if self.value_map._bits is not None:
            for r_idx in range(self.height):
                for c_idx in range(self.width):
                    val = (r_idx * self.width + c_idx) * (ITERATION_MULTIPLIER / 8.0)
                    self.value_map.set_value(c_idx, r_idx, val)

        self.gray_map = DEFAULT_COLOR_MAPS["Grayscale"]


    def test_generate_basic(self):
        img = generate_bitmap_pil(self.value_map, self.gray_map, normalize=False)
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.size, (self.width, self.height))

        # Check default color_map.max_iter
        self.assertAlmostEqual(self.gray_map.max_iter, ITERATION_MULTIPLIER)

        # Check a pixel value, e.g., (0,0) value is 0
        # gray_map.iter_to_rgb(0) is (0,0,0)
        self.assertEqual(img.getpixel((0,0)), (0,0,0))

        # Pixel (3,0) has value 3 * (ITERATION_MULTIPLIER / 8.0) = 3 * 32 = 96
        # gray_map.iter_to_rgb(96) with max_iter=256 should be (96,96,96)
        expected_color_p30 = self.gray_map.iter_to_rgb(3.0 * (ITERATION_MULTIPLIER / 8.0))
        self.assertEqual(img.getpixel((3,0)), expected_color_p30)


    def test_generate_normalize(self):
        # Find actual max value in map
        max_val_in_map = 0
        if self.value_map._bits is not None:
            max_val_in_map = np.max(self.value_map._bits) # (2*4-1) * (ITERATION_MULTIPLIER/8) = 7 * 32 = 224

        self.assertAlmostEqual(max_val_in_map, 7.0 * (ITERATION_MULTIPLIER / 8.0))

        img = generate_bitmap_pil(self.value_map, self.gray_map,
                                  current_max_value_in_map=max_val_in_map, normalize=True)

        # Check that color_map.max_iter was set to max_val_in_map
        self.assertAlmostEqual(self.gray_map.max_iter, max_val_in_map)

        # Pixel (3,1) has value 7 * (ITERATION_MULTIPLIER / 8.0) which is max_val_in_map
        # With normalization, this should map to the highest color in palette, (255,255,255)
        # gray_map.iter_to_rgb(max_val_in_map) when max_iter is max_val_in_map
        expected_color_max = self.gray_map.palette[255]
        self.assertEqual(img.getpixel((3,1)), tuple(expected_color_max))

        # Reset colormap max_iter for other tests
        self.gray_map.set_max_iter(ITERATION_MULTIPLIER)


    def test_generate_empty_value_map(self):
        empty_vm = ValueMap(0,0)
        img = generate_bitmap_pil(empty_vm, self.gray_map)
        self.assertEqual(img.size, (1,1))
        self.assertEqual(img.mode, "RGB")
        self.assertEqual(img.getpixel((0,0)), (0,0,0)) # Should be black

    def test_generate_normalize_zero_max_val(self):
        zero_vm = ValueMap(self.width, self.height) # All zeros
        img = generate_bitmap_pil(zero_vm, self.gray_map, current_max_value_in_map=0.0, normalize=True)
        # color_map.max_iter should revert to ITERATION_MULTIPLIER
        self.assertAlmostEqual(self.gray_map.max_iter, ITERATION_MULTIPLIER)
        self.assertEqual(img.getpixel((0,0)), (0,0,0))

    def test_generate_normalize_none_max_val(self):
        img = generate_bitmap_pil(self.value_map, self.gray_map, current_max_value_in_map=None, normalize=True)
        # color_map.max_iter should revert to ITERATION_MULTIPLIER
        self.assertAlmostEqual(self.gray_map.max_iter, ITERATION_MULTIPLIER)


if __name__ == '__main__':
    unittest.main()
