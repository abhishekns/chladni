import unittest
import numpy as np
import math
import threading # For stop_event in calculate_chladni_pattern tests

# Adjust import path for tests
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chladni.core import ValueMap, WaveInfo, calculate_chladni_pattern, MIN_FREQ_RATIO, ITERATION_MULTIPLIER

class TestValueMap(unittest.TestCase):

    def test_initialization(self):
        vm = ValueMap(10, 20)
        self.assertEqual(vm.width, 10)
        self.assertEqual(vm.height, 20)
        self.assertIsNotNone(vm._bits)
        self.assertEqual(vm._bits.shape, (20, 10))
        self.assertEqual(vm._bits.dtype, np.float32)
        self.assertTrue(np.all(vm._bits == 0))

        vm_zero = ValueMap(0, 0)
        self.assertEqual(vm_zero.width, 0)
        self.assertEqual(vm_zero.height, 0)
        self.assertIsNone(vm_zero._bits) # _bits is None for zero size
        self.assertTrue(vm_zero.empty)

        vm_zero_w = ValueMap(0, 10)
        self.assertEqual(vm_zero_w.width, 0)
        self.assertEqual(vm_zero_w.height, 10)
        self.assertIsNone(vm_zero_w._bits)
        self.assertTrue(vm_zero_w.empty)

        vm_zero_h = ValueMap(10, 0)
        self.assertEqual(vm_zero_h.width, 10)
        self.assertEqual(vm_zero_h.height, 0)
        self.assertIsNone(vm_zero_h._bits)
        self.assertTrue(vm_zero_h.empty)

        # Test with negative inputs, should be clamped to 0
        vm_neg = ValueMap(-5, -10)
        self.assertEqual(vm_neg.width, 0)
        self.assertEqual(vm_neg.height, 0)
        self.assertIsNone(vm_neg._bits)


    def test_width_height_properties(self):
        vm = ValueMap(5, 8)
        self.assertEqual(vm.width, 5)
        self.assertEqual(vm.height, 8)

        vm.width = 12
        self.assertEqual(vm.width, 12)
        self.assertEqual(vm.height, 8) # Height should remain unchanged
        self.assertEqual(vm._bits.shape, (8, 12))
        self.assertTrue(np.all(vm._bits == 0)) # Should be new zeroed array

        vm.height = 15
        self.assertEqual(vm.width, 12) # Width should remain unchanged
        self.assertEqual(vm.height, 15)
        self.assertEqual(vm._bits.shape, (15, 12))
        self.assertTrue(np.all(vm._bits == 0))

    def test_set_size(self):
        vm = ValueMap(3, 4)
        vm._bits.fill(1.0) # Fill with some data

        # Resize to larger
        changed = vm.set_size(5, 6)
        self.assertTrue(changed)
        self.assertEqual(vm.width, 5)
        self.assertEqual(vm.height, 6)
        self.assertEqual(vm._bits.shape, (6, 5))
        self.assertTrue(np.all(vm._bits == 0)) # Data is not preserved, new array is zeroed

        # Resize to smaller
        vm._bits.fill(2.0)
        changed = vm.set_size(2, 3)
        self.assertTrue(changed)
        self.assertEqual(vm.width, 2)
        self.assertEqual(vm.height, 3)
        self.assertEqual(vm._bits.shape, (3, 2))
        self.assertTrue(np.all(vm._bits == 0))

        # Resize to same dimensions
        vm._bits.fill(3.0)
        changed = vm.set_size(2, 3)
        self.assertFalse(changed) # No change in size
        self.assertTrue(np.all(vm._bits == 3.0)) # Data should be preserved if no actual change

        # Resize to zero
        changed = vm.set_size(0, 0)
        self.assertTrue(changed)
        self.assertEqual(vm.width, 0)
        self.assertEqual(vm.height, 0)
        self.assertIsNone(vm._bits)
        self.assertTrue(vm.empty)

        # Resize from zero
        changed = vm.set_size(2,2)
        self.assertTrue(changed)
        self.assertEqual(vm.width, 2)
        self.assertEqual(vm.height, 2)
        self.assertIsNotNone(vm._bits)


    def test_get_set_value(self):
        vm = ValueMap(3, 2)
        vm.set_value(0, 0, 1.1)
        vm.set_value(2, 1, 2.2)
        self.assertAlmostEqual(vm.get_value(0, 0), 1.1, places=5)
        self.assertAlmostEqual(vm.get_value(2, 1), 2.2, places=5)

        # Test out of bounds
        with self.assertRaises(IndexError):
            vm.get_value(3, 0)
        with self.assertRaises(IndexError):
            vm.set_value(0, 2, 3.3)
        with self.assertRaises(IndexError):
            vm.get_value(-1,0)
        with self.assertRaises(IndexError):
            vm.set_value(0,-1, 3.3)


        vm_zero = ValueMap(0,0)
        with self.assertRaises(ValueError): # _bits is None
            vm_zero.get_value(0,0)
        with self.assertRaises(ValueError):
            vm_zero.set_value(0,0,1.0)


    def test_getitem_setitem(self):
        vm = ValueMap(3, 2)
        vm[0, 0] = 1.1 # (y, x) indexing for __setitem__ matching numpy
        vm[1, 2] = 2.2
        self.assertAlmostEqual(vm[0, 0], 1.1, places=5)
        self.assertAlmostEqual(vm[1, 2], 2.2, places=5)

        # Test out of bounds
        with self.assertRaises(IndexError):
            _ = vm[0, 3]
        with self.assertRaises(IndexError):
            vm[2, 0] = 3.3
        with self.assertRaises(IndexError):
            _ = vm[-1,0]
        with self.assertRaises(IndexError):
            vm[0,-1] = 3.3


        with self.assertRaises(TypeError):
            _ = vm[0]
        with self.assertRaises(TypeError):
            vm[0] = 1.0

        vm_zero = ValueMap(0,0)
        with self.assertRaises(ValueError):
            _ = vm_zero[0,0]
        with self.assertRaises(ValueError):
            vm_zero[0,0] = 1.0


    def test_get_scanline(self):
        vm = ValueMap(4, 3)
        for x in range(4):
            vm.set_value(x, 1, float(x + 0.5)) # Set values for row 1

        scanline1 = vm.get_scanline(1)
        self.assertIsInstance(scanline1, np.ndarray)
        self.assertEqual(scanline1.shape, (4,))
        expected = np.array([0.5, 1.5, 2.5, 3.5], dtype=np.float32)
        self.assertTrue(np.allclose(scanline1, expected))

        # Test out of bounds
        with self.assertRaises(IndexError):
            vm.get_scanline(3)
        with self.assertRaises(IndexError):
            vm.get_scanline(-1)

        vm_zero = ValueMap(0,0)
        with self.assertRaises(ValueError):
            vm_zero.get_scanline(0)


    def test_clear(self):
        vm = ValueMap(2, 2)
        vm.set_value(0, 0, 1.0)
        vm.set_value(1, 1, 2.0)
        vm.clear()
        self.assertTrue(np.all(vm._bits == 0))

        vm_zero = ValueMap(0,0)
        vm_zero.clear() # Should not raise error
        self.assertTrue(vm_zero.empty)


    def test_delete(self):
        vm = ValueMap(2, 2)
        vm.delete()
        self.assertEqual(vm.width, 0)
        self.assertEqual(vm.height, 0)
        self.assertIsNone(vm._bits)
        self.assertTrue(vm.empty)

        vm_already_zero = ValueMap(0,0)
        vm_already_zero.delete() # Should not raise error
        self.assertTrue(vm_already_zero.empty)

    def test_empty_property(self):
        vm = ValueMap(1, 1)
        self.assertFalse(vm.empty)
        vm.delete()
        self.assertTrue(vm.empty)

        vm_zero_init = ValueMap(0, 0)
        self.assertTrue(vm_zero_init.empty)
        vm_zero_init.set_size(1,1)
        self.assertFalse(vm_zero_init.empty)


class TestCalculateChladniPattern(unittest.TestCase):

    def setUp(self):
        self.width = 10 # Small for faster tests
        self.height = 10
        self.value_map = ValueMap(self.width, self.height)
        self.waves = [
            WaveInfo(on=True, amplitude=1.0, frequency=2.0, phase=0.0),
            WaveInfo(on=True, amplitude=0.5, frequency=3.0, phase=90.0)
        ]

    def test_basic_calculation(self):
        max_val = calculate_chladni_pattern(self.value_map, self.waves, self.width, self.height)
        self.assertGreater(max_val, 0) # Expect some pattern
        self.assertFalse(np.all(self.value_map._bits == 0)) # Map should be filled

        # Check specific value (center pixel) - this is a bit of a "golden value" test
        # For (x=5, y=5) in a 10x10 grid (0-indexed), rx = 5/10 = 0.5, ry = 5/10 = 0.5
        # Wave 1: A=1, f=2, p=0. q = 2*pi*2 = 4*pi. phase_rad = 0.
        #   term_x1 = cos(4*pi*0.5 + 0) = cos(2*pi) = 1
        #   term_y1 = cos(4*pi*0.5 + 0) = cos(2*pi) = 1
        #   v1 = 1.0 * 1 * 1 = 1.0
        # Wave 2: A=0.5, f=3, p=90. q = 2*pi*3 = 6*pi. phase_rad = pi/2.
        #   term_x2 = cos(6*pi*0.5 + pi/2) = cos(3*pi + pi/2) = cos(3.5*pi) = 0
        #   term_y2 = cos(6*pi*0.5 + pi/2) = cos(3.5*pi) = 0
        #   v2 = 0.5 * 0 * 0 = 0
        # Total v = 1.0 + 0 = 1.0
        # current_val = abs(1.0) * ITERATION_MULTIPLIER
        expected_center_val = abs(1.0) * ITERATION_MULTIPLIER
        # Note: indexing for value_map.get_value is (x, y), for _bits is (y,x)
        center_x, center_y = self.width // 2, self.height // 2
        # If width/height are even, rx,ry for center_x, center_y will be (size/2)/size = 0.5
        # For ValueMap(10,10), center_x=5, center_y=5.
        # rx = 5/10 = 0.5, ry = 5/10 = 0.5.
        # This matches the manual calculation.
        self.assertAlmostEqual(self.value_map.get_value(center_x, center_y), expected_center_val, places=5)


    def test_no_active_waves(self):
        waves_off = [
            WaveInfo(on=False, amplitude=1.0, frequency=2.0, phase=0.0),
            WaveInfo(on=False, amplitude=0.5, frequency=3.0, phase=90.0)
        ]
        max_val = calculate_chladni_pattern(self.value_map, waves_off, self.width, self.height)
        self.assertEqual(max_val, 0)
        self.assertTrue(np.all(self.value_map._bits == 0))

    def test_empty_wave_list(self):
        max_val = calculate_chladni_pattern(self.value_map, [], self.width, self.height)
        self.assertEqual(max_val, 0)
        self.assertTrue(np.all(self.value_map._bits == 0))

    def test_resize_value_map(self):
        # ValueMap is 10x10, calculate for 5x5
        new_width, new_height = 5, 5
        calculate_chladni_pattern(self.value_map, self.waves, new_width, new_height)
        self.assertEqual(self.value_map.width, new_width)
        self.assertEqual(self.value_map.height, new_height)
        self.assertFalse(np.all(self.value_map._bits == 0))

    def test_clear_value_map_before_calc(self):
        self.value_map._bits.fill(123.45) # Pre-fill with some data
        calculate_chladni_pattern(self.value_map, self.waves, self.width, self.height)
        # Ensure the pre-filled data is gone and new pattern is there
        # The test_basic_calculation already verifies a specific point,
        # so just ensuring it's not all 123.45 or all 0 is enough here.
        self.assertFalse(np.all(self.value_map._bits == 123.45))
        self.assertFalse(np.all(self.value_map._bits == 0))


    def test_stop_event(self):
        stop_event = threading.Event()

        # Test immediate stop
        stop_event.set()
        max_val = calculate_chladni_pattern(self.value_map, self.waves, self.width, self.height, stop_event)
        # Depending on when the check happens, some values might be calculated for the first row/pixel
        # For a very small map, it might complete. For a larger one, it should stop early.
        # If it stops before any calculation, max_val should be 0 and map all zeros.
        # Let's check if it's different from a full calculation
        full_calc_map = ValueMap(self.width, self.height)
        full_max_val = calculate_chladni_pattern(full_calc_map, self.waves, self.width, self.height, None)

        # If stopped very early, max_val might be 0.
        # If it ran for a bit, it might be less than full_max_val.
        # And the map content would differ.
        self.assertNotEqual(np.sum(self.value_map._bits), np.sum(full_calc_map._bits), "Pattern should differ if stopped early.")
        self.assertLessEqual(max_val, full_max_val) # Max value should be less or equal if stopped early

    def test_min_freq_ratio(self):
        waves_low_freq = [
            WaveInfo(on=True, amplitude=1.0, frequency=MIN_FREQ_RATIO - 0.01, phase=0.0)
        ]
        max_val = calculate_chladni_pattern(self.value_map, waves_low_freq, self.width, self.height)
        self.assertEqual(max_val, 0) # Frequency is below minimum, wave should be skipped
        self.assertTrue(np.all(self.value_map._bits == 0))

        waves_at_min_freq = [
            WaveInfo(on=True, amplitude=1.0, frequency=MIN_FREQ_RATIO, phase=0.0)
        ]
        max_val_at_min = calculate_chladni_pattern(self.value_map, waves_at_min_freq, self.width, self.height)
        self.assertGreater(max_val_at_min, 0) # Frequency is at minimum, wave should be processed

    def test_iteration_multiplier_application(self):
        # Use a single wave, simple parameters, check one point
        wave = [WaveInfo(on=True, amplitude=1.0, frequency=2.0, phase=0.0)] # Same as basic_calculation wave 1
        # For x=width/2, y=height/2, v_before_abs_mult = 1.0
        # Expected value = abs(1.0) * ITERATION_MULTIPLIER

        calculate_chladni_pattern(self.value_map, wave, self.width, self.height)

        center_x, center_y = self.width // 2, self.height // 2
        # Manually calculate v for this point:
        rx = center_x / self.width
        ry = center_y / self.height
        phase_rad = math.radians(wave[0].phase)
        q = 2 * math.pi * wave[0].frequency
        term_x = math.cos(q * rx + phase_rad)
        term_y = math.cos(q * ry + phase_rad)
        v_manual = wave[0].amplitude * term_x * term_y

        expected_val_at_point = abs(v_manual) * ITERATION_MULTIPLIER

        self.assertAlmostEqual(self.value_map.get_value(center_x, center_y), expected_val_at_point, places=5)


if __name__ == '__main__':
    unittest.main()
