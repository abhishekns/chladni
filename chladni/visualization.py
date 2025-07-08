import math
from typing import Tuple, List
from PIL import Image
import numpy as np

from .core import ValueMap, ITERATION_MULTIPLIER

# Type alias for color
RGBColor = Tuple[int, int, int]

def _mix_colors(color1: RGBColor, color2: RGBColor, weight2_percent: int) -> RGBColor:
    """
    Interpolates between two colors.
    weight2_percent: 0 means color1, 100 means color2.
    This matches the logic: MixColors(fPalette[i + 1], fPalette[i], w)
    where w is for palette[i+1] and (100-w) is for palette[i].
    So color1 is palette[i+1] (target), color2 is palette[i] (source).
    """
    w2 = weight2_percent / 100.0
    w1 = 1.0 - w2
    r = int(color2[0] * w1 + color1[0] * w2)
    g = int(color2[1] * w1 + color1[1] * w2)
    b = int(color2[2] * w1 + color1[2] * w2)
    return (
        max(0, min(255, r)),
        max(0, min(255, g)),
        max(0, min(255, b))
    )

class ColorMap:
    def __init__(self, name: str, palette: List[RGBColor], max_iter: float = ITERATION_MULTIPLIER):
        if len(palette) != 256:
            raise ValueError("Palette must contain 256 colors.")
        self.name = name
        self.palette: np.ndarray = np.array(palette, dtype=np.uint8) # Store as numpy array for easier slicing
        self.max_iter = float(max_iter)
        self._rec_max_col: float = 0.0
        self._update_calc()

    def _update_calc(self):
        if self.max_iter > 0:
            self._rec_max_col = 1.0 / self.max_iter
        else:
            self._rec_max_col = 0.0
        # fOffs related to fOffset and fRepeating not implemented yet,
        # as their direct use in IterToACColor was not clear.

    def set_max_iter(self, new_max_iter: float):
        if self.max_iter == new_max_iter:
            return
        self.max_iter = float(new_max_iter)
        self._update_calc()

    def iter_to_rgb(self, value: float) -> RGBColor:
        if value >= self.max_iter:
            return tuple(self.palette[255])
        if value <= 0:
            return tuple(self.palette[0])

        # n_norm = Frac(Iter * fRecMaxCol) -> maps Iter to [0,1) if Iter is within [0, MaxIter)
        # Python's math.fmod(x, 1.0) is similar to Frac if x is positive.
        # Or (x * y) - floor(x * y)
        normalized_value = (value * self._rec_max_col)
        # The original code uses Frac. If Iter * fRecMaxCol can be > 1 (due to Repeating?),
        # then Frac is important. For now, assume it stays within [0,1) for one cycle.
        # If fRepeating is > 0, the behavior might be more complex.
        # Let's assume simple mapping for now: value is scaled to [0,1) then to [0,255)

        n_scaled = normalized_value * 255.0

        idx_float = np.clip(n_scaled, 0, 255) # Ensure it's within [0, 255] range

        i = int(idx_float) # Convert numpy float to int for truncation

        # Ensure i is a valid index, especially for i=255
        i = min(i, 254) # If i becomes 255, i+1 would be out of bounds for palette

        fractional_part = idx_float - i

        # w := Round(n * K_PERCENT); K_PERCENT = 100
        # This weight is for color_i_plus_1
        weight_for_color_i_plus_1 = round(fractional_part * 100.0)
        weight_for_color_i_plus_1 = max(0, min(100, weight_for_color_i_plus_1))

        color_i = tuple(self.palette[i])
        color_i_plus_1 = tuple(self.palette[i + 1])

        return _mix_colors(color_i_plus_1, color_i, weight_for_color_i_plus_1)

# --- Predefined Palettes ---
def create_grayscale_palette() -> List[RGBColor]:
    return [(i, i, i) for i in range(256)]

def create_spectrum_palette() -> List[RGBColor]:
    palette = []
    for i in range(256):
        r, g, b = 0, 0, 0
        if i < 64: # Blue to Cyan
            b = 255
            g = i * 4
        elif i < 128: # Cyan to Green
            g = 255
            b = 255 - (i - 64) * 4
        elif i < 192: # Green to Yellow
            g = 255
            r = (i - 128) * 4
        else: # Yellow to Red
            r = 255
            g = 255 - (i - 192) * 4
        palette.append((r,g,b))
    return palette

DEFAULT_COLOR_MAPS = {
    "Grayscale": ColorMap("Grayscale", create_grayscale_palette()),
    "Spectrum": ColorMap("Spectrum", create_spectrum_palette()),
}

def generate_bitmap_pil(
    value_map: ValueMap,
    color_map: ColorMap,
    current_max_value_in_map: float | None = None, # The actual max value from ValueMap if normalization is desired
    normalize: bool = False
) -> Image.Image:
    if value_map.width == 0 or value_map.height == 0:
        return Image.new('RGB', (1,1), color=(0,0,0)) # Return a dummy 1x1 image

    img = Image.new('RGB', (value_map.width, value_map.height))
    pixels = img.load()

    # Set the color map's iteration limit
    if normalize and current_max_value_in_map is not None and current_max_value_in_map > 0:
        color_map.set_max_iter(current_max_value_in_map)
    else:
        # Default behavior from Pascal: make_bitmap sets MaxIter to ITERATION_MULTIPLIER
        # if not normalizing or fValMax is 0
        color_map.set_max_iter(ITERATION_MULTIPLIER)

    for y_idx in range(value_map.height):
        for x_idx in range(value_map.width):
            val = value_map.get_value(x_idx, y_idx)
            pixels[x_idx, y_idx] = color_map.iter_to_rgb(val)

    return img

if __name__ == '__main__':
    print("Testing ColorMap and Bitmap Generation...")

    # Test Grayscale
    gray_map = DEFAULT_COLOR_MAPS["Grayscale"]
    gray_map.set_max_iter(255) # For simple 0-255 mapping
    print(f"Grayscale(0): {gray_map.iter_to_rgb(0)}")
    print(f"Grayscale(127.5): {gray_map.iter_to_rgb(127.5)}") # Should be (127,127,127) or (128,128,128)
    print(f"Grayscale(255): {gray_map.iter_to_rgb(255)}")

    # Test Spectrum
    spectrum_map = DEFAULT_COLOR_MAPS["Spectrum"]
    spectrum_map.set_max_iter(255)
    print(f"Spectrum(0): {spectrum_map.iter_to_rgb(0)}")          # Expected: Blueish
    print(f"Spectrum(63.9): {spectrum_map.iter_to_rgb(63.9)}")    # Expected: Cyanish
    print(f"Spectrum(127.9): {spectrum_map.iter_to_rgb(127.9)}")  # Expected: greenish
    print(f"Spectrum(191.9): {spectrum_map.iter_to_rgb(191.9)}")  # Expected: yellowish
    print(f"Spectrum(255): {spectrum_map.iter_to_rgb(255)}")    # Expected: Reddish

    # Test Bitmap Generation
    test_vm_width, test_vm_height = 64, 32
    test_value_map = ValueMap(test_vm_width, test_vm_height)

    # Create a gradient in the value map
    max_val_for_grad = 0
    if test_value_map._bits is not None:
        for r in range(test_vm_height):
            for c in range(test_vm_width):
                # Simple gradient from 0 to nearly ITERATION_MULTIPLIER
                val = (c / (test_vm_width -1)) * ITERATION_MULTIPLIER * 0.95
                test_value_map.set_value(c, r, val)
                if val > max_val_for_grad: max_val_for_grad = val

    print(f"Max value in test_value_map for gradient: {max_val_for_grad}")

    # Generate with default ITERATION_MULTIPLIER limit
    print("Generating bitmap with Spectrum (default max_iter)...")
    img_spectrum = generate_bitmap_pil(test_value_map, spectrum_map)
    img_spectrum.save("test_spectrum_default.png")
    print("Saved test_spectrum_default.png")

    # Generate with normalization
    print("Generating bitmap with Spectrum (normalized)...")
    img_spectrum_norm = generate_bitmap_pil(test_value_map, spectrum_map, current_max_value_in_map=max_val_for_grad, normalize=True)
    img_spectrum_norm.save("test_spectrum_normalized.png")
    print("Saved test_spectrum_normalized.png (should look more full-range than default if max_val_for_grad < ITERATION_MULTIPLIER)")

    # Test with Grayscale
    print("Generating bitmap with Grayscale (normalized)...")
    img_gray_norm = generate_bitmap_pil(test_value_map, gray_map, current_max_value_in_map=max_val_for_grad, normalize=True)
    img_gray_norm.save("test_gray_normalized.png")
    print("Saved test_gray_normalized.png")

    print("Visualization tests completed. Check the generated PNG files.")
