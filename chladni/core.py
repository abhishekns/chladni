import numpy as np
from dataclasses import dataclass

# Constants from ucladni.pas
MIN_AMPLITUDE = -1.0
MAX_AMPLITUDE = 1.0
MIN_ANGLE = -360.0
MAX_ANGLE = 360.0
ITERATION_MULTIPLIER = 256.0  # In Pascal it's an integer, but used in float calcs

# Global variables in ucladni.pas, treat as configurable constants
MIN_FREQ_RATIO = 0.1
MAX_FREQ_RATIO = 20.0

@dataclass
class WaveInfo:
    on: bool = False
    amplitude: float = 0.0
    frequency: float = 0.0
    phase: float = 0.0

class ValueMap:
    def __init__(self, width: int, height: int):
        self._width: int = 0
        self._height: int = 0
        self._bits: np.ndarray | None = None
        # fOnChange: TNotifyEvent - GUI related, skip for now
        # fOnResize: TNotifyEvent - GUI related, skip for now
        self.set_size(width, height)

    @property
    def width(self) -> int:
        return self._width

    @width.setter
    def width(self, new_width: int) -> None:
        self.set_size(new_width, self._height)

    @property
    def height(self) -> int:
        return self._height

    @height.setter
    def height(self, new_height: int) -> None:
        self.set_size(self._width, new_height)

    def _change_size(self, new_width: int, new_height: int) -> None:
        self._width = new_width
        self._height = new_height
        if self._width == 0 or self._height == 0:
            self._bits = None
        else:
            # In Pascal, ReallocMem preserves content if possible.
            # NumPy array creation of a new size does not preserve.
            # If preservation is needed, a more complex copy would be required.
            # For now, assume new size means new array.
            self._bits = np.zeros((new_height, new_width), dtype=np.float32)

    def get_value(self, x: int, y: int) -> float:
        if self._bits is None:
            raise ValueError("ValueMap is not initialized or has zero size")
        if not (0 <= y < self._height and 0 <= x < self._width):
            raise IndexError("Coordinates out of bounds")
        return self._bits[y, x]

    def set_value(self, x: int, y: int, value: float) -> None:
        if self._bits is None:
            raise ValueError("ValueMap is not initialized or has zero size")
        if not (0 <= y < self._height and 0 <= x < self._width):
            raise IndexError("Coordinates out of bounds")
        self._bits[y, x] = value
        # self.changed() # GUI related

    def get_scanline(self, y: int) -> np.ndarray:
        if self._bits is None:
            raise ValueError("ValueMap is not initialized or has zero size")
        if not (0 <= y < self._height):
            raise IndexError("Y coordinate out of bounds")
        return self._bits[y, :]

    def delete(self) -> None:
        self.set_size(0, 0)

    # def changed(self) -> None:
    #     # if self.fOnChange: self.fOnChange(self) # GUI related
    #     pass

    def clear(self) -> None:
        if self._bits is not None:
            self._bits.fill(0.0)
        # self.changed() # GUI related

    # def resized(self) -> None:
    #     # if self.fOnResize: self.fOnResize(self) # GUI related
    #     pass

    @property
    def empty(self) -> bool:
        return self._bits is None or self._bits.size == 0

    def set_size(self, new_width: int, new_height: int) -> bool:
        new_width = max(0, new_width)
        new_height = max(0, new_height)

        changed = (new_width != self._width) or (new_height != self._height)
        if changed:
            self._change_size(new_width, new_height)
            # self.changed() # GUI related
            # self.resized() # GUI related
        return changed

    # Pythonic way to access elements: map[y, x] or map.get_value(x,y) / map.set_value(x,y)
    def __getitem__(self, key):
        if isinstance(key, tuple) and len(key) == 2:
            y, x = key
            if self._bits is None:
                 raise ValueError("ValueMap is not initialized or has zero size")
            if not (0 <= y < self._height and 0 <= x < self._width):
                raise IndexError("Coordinates out of bounds")
            return self._bits[y, x]
        raise TypeError("Invalid index type. Use [row, col].")

    def __setitem__(self, key, value):
        if isinstance(key, tuple) and len(key) == 2:
            y, x = key
            if self._bits is None:
                 raise ValueError("ValueMap is not initialized or has zero size")
            if not (0 <= y < self._height and 0 <= x < self._width):
                raise IndexError("Coordinates out of bounds")
            self._bits[y, x] = value
            # self.changed() # GUI related
        else:
            raise TypeError("Invalid index type. Use [row, col].")

if __name__ == '__main__':
    # Basic test
    vm = ValueMap(10, 5)
    print(f"Initial size: {vm.width}x{vm.height}")
    vm.set_value(0, 0, 1.5)
    vm.set_value(9, 4, 2.5)
    print(f"Value at (0,0): {vm.get_value(0,0)}")
    print(f"Value at (9,4) using __getitem__: {vm[4,9]}") # Note: NumPy is row-major (y,x)

    vm[4,9] = 3.5 # Using __setitem__
    print(f"Value at (9,4) after __setitem__: {vm[4,9]}")

    scanline = vm.get_scanline(4)
    print(f"Scanline 4: {scanline}")

    vm.clear()
    print(f"Value at (9,4) after clear: {vm[4,9]}")

    vm.width = 20
    print(f"New size: {vm.width}x{vm.height}")
    print(f"Is empty: {vm.empty}")

    vm.delete()
    print(f"Size after delete: {vm.width}x{vm.height}")
    print(f"Is empty after delete: {vm.empty}")

    # Test initialization with 0 size
    vm_zero = ValueMap(0,0)
    print(f"vm_zero empty: {vm_zero.empty}")
    try:
        vm_zero.set_value(0,0,1.0)
    except ValueError as e:
        print(f"Error setting value on zero-sized map: {e}")

    vm_zero.set_size(2,2)
    print(f"vm_zero empty after resize: {vm_zero.empty}")
    vm_zero[0,0] = 1.23
    print(f"vm_zero[0,0]: {vm_zero[0,0]}")


import math
import threading # For threading.Event

def calculate_chladni_pattern(
    value_map: ValueMap,
    wave_infos: list[WaveInfo],
    width: int,
    height: int,
    stop_event: threading.Event | None = None # Optional for non-threaded use
) -> float:
    """
    Calculates the Chladni pattern and stores it in value_map.
    Returns the maximum absolute value found during calculation (for normalization).
    """
    if value_map.width != width or value_map.height != height:
        value_map.set_size(width, height)
    else:
        value_map.clear()

    r_width = 1.0 / width
    r_height = 1.0 / height
    val_max = 0.0

    for y in range(height):
        if stop_event and stop_event.is_set(): break # Check before starting a row
        ry = y * r_height
        for x in range(width):
            if stop_event and x % 64 == 0 and stop_event.is_set(): break # Check periodically within a row

            v = 0.0
            rx = x * r_width
            for info in wave_infos: # This loop is usually fast, might not need a check here
                if not info.on or info.frequency < MIN_FREQ_RATIO:
                    continue

                phase_rad = math.radians(info.phase)
                q = 2 * math.pi * info.frequency

                # Original formula: v += info.Amplitude * cos(q * rx + p) * cos(q * ry + p);
                # The Pascal code seems to apply phase to both terms.
                # It's common in Chladni patterns to see terms like:
                # cos(n*pi*x/L + phase_x) * cos(m*pi*y/L + phase_y) or variations.
                # The original code uses the same 'q' (related to frequency) and 'p' (phase) for both x and y.
                # Let's stick to the original formula structure.
                term_x = math.cos(q * rx + phase_rad)
                term_y = math.cos(q * ry + phase_rad)
                v += info.amplitude * term_x * term_y

            # In Pascal: v := Abs(v) * ITERATION_MULTIPLIER;
            # ITERATION_MULTIPLIER seems to be a scaling factor.
            # Let's apply it before taking Abs for now, then val_max will be based on scaled values.
            # No, the original takes Abs then multiplies.

            current_val = abs(v) * ITERATION_MULTIPLIER
            value_map.set_value(x, y, current_val)
            if current_val > val_max:
                val_max = current_val

        if stop_event and stop_event.is_set(): break # Check after finishing a row

    return val_max

if __name__ == '__main__':
    # Basic test for ValueMap
    vm = ValueMap(10, 5)
    print(f"Initial size: {vm.width}x{vm.height}")
    vm.set_value(0, 0, 1.5)
    vm.set_value(9, 4, 2.5)
    print(f"Value at (0,0): {vm.get_value(0,0)}")
    print(f"Value at (9,4) using __getitem__: {vm[4,9]}")

    vm[4,9] = 3.5
    print(f"Value at (9,4) after __setitem__: {vm[4,9]}")

    scanline = vm.get_scanline(4)
    print(f"Scanline 4: {scanline}")

    vm.clear()
    print(f"Value at (9,4) after clear: {vm[4,9]}")

    vm.width = 20
    print(f"New size: {vm.width}x{vm.height}")
    print(f"Is empty: {vm.empty}")

    vm.delete()
    print(f"Size after delete: {vm.width}x{vm.height}")
    print(f"Is empty after delete: {vm.empty}")

    vm_zero = ValueMap(0,0)
    print(f"vm_zero empty: {vm_zero.empty}")
    try:
        vm_zero.set_value(0,0,1.0)
    except ValueError as e:
        print(f"Error setting value on zero-sized map: {e}")

    vm_zero.set_size(2,2)
    print(f"vm_zero empty after resize: {vm_zero.empty}")
    vm_zero[0,0] = 1.23
    print(f"vm_zero[0,0]: {vm_zero[0,0]}")

    # Test Chladni calculation
    print("\nTesting Chladni calculation...")
    test_width, test_height = 64, 64 # Small size for quick test
    test_vm = ValueMap(test_width, test_height)

    # Example wave infos (mimicking a simple pattern)
    # Values taken from one of the .chl examples (Test1.chl, first few active waves)
    # On; Amplitude; Frequency; Phase
    # 1;0.1719;3.1;-240.2
    # 1;0.8633;5;-233.4
    # 1;-0.25;7.9;311.6

    # Simpler test case first: one wave
    waves = [
        WaveInfo(on=True, amplitude=1.0, frequency=4.0, phase=0.0),
        WaveInfo(on=True, amplitude=0.5, frequency=6.0, phase=90.0),
    ]

    max_val = calculate_chladni_pattern(test_vm, waves, test_width, test_height)
    print(f"Calculation complete. Max value: {max_val}")
    print(f"Value at (0,0): {test_vm[0,0]}")
    print(f"Value at (32,32): {test_vm[32,32]}")

    # A quick check, sum of values (not a rigorous test)
    total_sum = np.sum(test_vm._bits) # Accessing _bits directly for testing sum
    print(f"Sum of all values in test_vm: {total_sum}")

    # Test with no active waves
    waves_off = [WaveInfo(on=False, amplitude=1.0, frequency=4.0, phase=0.0)]
    max_val_off = calculate_chladni_pattern(test_vm, waves_off, test_width, test_height)
    print(f"Calculation with all waves off. Max value: {max_val_off}")
    total_sum_off = np.sum(test_vm._bits)
    print(f"Sum of all values with waves off: {total_sum_off}")
    assert max_val_off == 0.0, "Max value should be 0 if all waves are off"
    assert total_sum_off == 0.0, "Total sum should be 0 if all waves are off"
    print("Chladni calculation test passed basic checks.")
