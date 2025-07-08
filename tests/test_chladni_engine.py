import unittest
import os
import shutil # For cleaning up test files/dirs
from PIL import Image
import numpy as np # For np.all() in test_file_save_no_level_map

# Adjust import path for tests
import sys
# This ensures that the 'src' and 'ui' directories are discoverable
# when running tests from the root of the 'chladni_py' project structure
# or if 'chladni_py' itself is in PYTHONPATH.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chladni.chladni_engine import ChladniSimulator, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_CAPACITY, CHL_UNTITLED, WaveInfo, DEFAULT_NORMALIZE, DEFAULT_MAP_NAME, MIN_AMPLITUDE, MAX_AMPLITUDE, MIN_FREQ_RATIO, MAX_FREQ_RATIO, MIN_ANGLE, MAX_ANGLE
from chladni.core import ValueMap, ITERATION_MULTIPLIER
from chladni.visualization import DEFAULT_COLOR_MAPS
import threading


class TestChladniSimulator(unittest.TestCase):

    def setUp(self):
        """Set up for each test."""
        self.simulator = ChladniSimulator()
        self.test_dir = "temp_engine_test_files" # Unique name for this test suite
        os.makedirs(self.test_dir, exist_ok=True)

    def tearDown(self):
        """Clean up after each test."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_initialization(self):
        self.assertEqual(self.simulator.width, DEFAULT_WIDTH)
        self.assertEqual(self.simulator.height, DEFAULT_HEIGHT)
        self.assertEqual(self.simulator.capacity, DEFAULT_CAPACITY)
        self.assertEqual(len(self.simulator.wave_infos), DEFAULT_CAPACITY)
        self.assertEqual(self.simulator.filename, CHL_UNTITLED)
        self.assertFalse(self.simulator.modified)
        self.assertIsNotNone(self.simulator._value_map)
        self.assertEqual(self.simulator._value_map.width, DEFAULT_WIDTH)
        self.assertEqual(self.simulator._value_map.height, DEFAULT_HEIGHT)
        self.assertEqual(self.simulator.normalize, DEFAULT_NORMALIZE)
        self.assertEqual(self.simulator.selected_color_map_name, DEFAULT_MAP_NAME)
        self.assertEqual(self.simulator._calculated_max_value, 0.0)

    def test_set_dimensions(self):
        self.simulator.set_dimensions(100, 150)
        self.assertEqual(self.simulator.width, 100)
        self.assertEqual(self.simulator.height, 150)
        self.assertEqual(self.simulator._value_map.width, 100)
        self.assertEqual(self.simulator._value_map.height, 150)
        self.assertTrue(self.simulator.modified)

        # Test dimension clamping
        self.simulator.set_dimensions(5, 5) # Below min
        self.assertEqual(self.simulator.width, 10)
        self.assertEqual(self.simulator.height, 10)

        self.simulator.set_dimensions(20000, 20000) # Above max
        self.assertEqual(self.simulator.width, 16384)
        self.assertEqual(self.simulator.height, 16384)

        # Test no change if dimensions are the same
        self.simulator.modified = False
        self.simulator.set_dimensions(16384, 16384)
        self.assertFalse(self.simulator.modified)


    def test_set_capacity(self):
        # initial_wave_info_objects = self.simulator.wave_infos # This was problematic as it's a reference

        initial_len = len(self.simulator.wave_infos) # Store initial length based on DEFAULT_CAPACITY
        self.assertEqual(initial_len, DEFAULT_CAPACITY)


        self.simulator.set_capacity(50)
        self.assertEqual(self.simulator.capacity, 50)
        self.assertEqual(len(self.simulator.wave_infos), 50)
        self.assertTrue(self.simulator.modified)
        # Check if new WaveInfo objects are added relative to original default
        self.assertGreater(len(self.simulator.wave_infos), initial_len)

        self.simulator.set_capacity(20)
        self.assertEqual(self.simulator.capacity, 20)
        self.assertEqual(len(self.simulator.wave_infos), 20)
        self.assertTrue(self.simulator.modified) # Should be true even when reducing

        # Test clamping to minimum 1
        self.simulator.set_capacity(0)
        self.assertEqual(self.simulator.capacity, 1)
        self.assertEqual(len(self.simulator.wave_infos), 1)

        # Test no change if capacity is the same
        self.simulator.modified = False
        self.simulator.set_capacity(1)
        self.assertFalse(self.simulator.modified)


    def test_reset(self):
        self.simulator.set_dimensions(100, 100)
        self.simulator.set_capacity(10)
        self.simulator.filename = "test.chl"
        self.simulator.modified = True
        self.simulator.normalize = False
        self.simulator.selected_color_map_name = "Grayscale"
        if self.simulator.wave_infos: # Ensure list is not empty
            self.simulator.wave_infos[0].amplitude = 1.0
        self.simulator._value_map.set_value(0,0,10.0)
        self.simulator._calculated_max_value = 10.0


        self.simulator.reset()

        self.assertEqual(self.simulator.width, DEFAULT_WIDTH)
        self.assertEqual(self.simulator.height, DEFAULT_HEIGHT)
        self.assertEqual(self.simulator.capacity, DEFAULT_CAPACITY)
        self.assertEqual(self.simulator.filename, CHL_UNTITLED)
        self.assertFalse(self.simulator.modified)
        self.assertEqual(self.simulator.normalize, DEFAULT_NORMALIZE)
        self.assertEqual(self.simulator.selected_color_map_name, DEFAULT_MAP_NAME)
        self.assertEqual(self.simulator._value_map.width, DEFAULT_WIDTH)
        self.assertEqual(self.simulator._value_map.height, DEFAULT_HEIGHT)
        self.assertTrue(np.all(self.simulator._value_map._bits == 0)) # Ensure map is cleared
        self.assertEqual(self.simulator._calculated_max_value, 0.0)

        if self.simulator.wave_infos:
             self.assertEqual(self.simulator.wave_infos[0].amplitude, 0.0) # Default WaveInfo state

    def test_recalculate_pattern_with_event(self):
        self.simulator.set_dimensions(50,50) # Smaller for faster test
        if self.simulator.wave_infos:
            self.simulator.wave_infos[0] = WaveInfo(on=True, amplitude=1.0, frequency=4.0, phase=0.0)

        self.simulator.recalculate_pattern_with_event(None)
        self.assertTrue(self.simulator.modified)
        self.assertGreater(self.simulator._calculated_max_value, 0)
        # Check if value map has non-zero data
        self.assertFalse(np.all(self.simulator._value_map._bits == 0))

        # Test with empty wave_infos (all off)
        self.simulator.wave_infos[0].on = False
        self.simulator.modified = False # Reset modified flag
        self.simulator.recalculate_pattern_with_event(None)
        self.assertTrue(self.simulator.modified) # Still modified as calculation ran
        self.assertEqual(self.simulator._calculated_max_value, 0)
        self.assertTrue(np.all(self.simulator._value_map._bits == 0))

        # Test stop event
        self.simulator.wave_infos[0].on = True
        stop_event = threading.Event()
        stop_event.set() # Stop immediately
        self.simulator.modified = False
        initial_max_val = self.simulator._calculated_max_value
        self.simulator.recalculate_pattern_with_event(stop_event)
        # If stopped early, modified might not be true, and max_value might not change or be partial.
        # The current implementation sets modified=True only if not stopped.
        self.assertFalse(self.simulator.modified)
        # _calculated_max_value might be non-zero if some calculation happened before stop
        # This depends on how quickly the stop_event is checked in calculate_chladni_pattern
        # For an immediate stop, it's likely 0 or unchanged.
        # For this test, let's assume it doesn't complete fully.
        # A more robust test would be to check if _value_map is partially filled or unchanged.


    def test_get_current_bitmap_pil_image(self):
        self.simulator.set_dimensions(20,20)
        self.simulator.wave_infos[0] = WaveInfo(on=True, amplitude=1.0, frequency=2.0, phase=0.0)
        self.simulator.recalculate_pattern_with_event(None)

        img_default = self.simulator.get_current_bitmap_pil_image()
        self.assertIsInstance(img_default, Image.Image)
        self.assertEqual(img_default.size, (20, 20))
        self.assertEqual(img_default.mode, "RGB")

        # Test with a different colormap
        original_map_name = self.simulator.selected_color_map_name
        self.simulator.selected_color_map_name = "Grayscale"
        img_gray = self.simulator.get_current_bitmap_pil_image()
        self.assertIsInstance(img_gray, Image.Image)
        # Reset colormap
        self.simulator.selected_color_map_name = original_map_name

        # Test normalization effect (indirectly by checking if colormap's max_iter is set)
        self.simulator.normalize = True
        self.simulator.recalculate_pattern_with_event(None) # Ensure _calculated_max_value is set
        calculated_max = self.simulator._calculated_max_value

        _ = self.simulator.get_current_bitmap_pil_image()
        active_color_map = self.simulator.get_current_color_map()
        self.assertAlmostEqual(active_color_map.max_iter, calculated_max, places=5)

        self.simulator.normalize = False
        _ = self.simulator.get_current_bitmap_pil_image()
        active_color_map = self.simulator.get_current_color_map()
        self.assertAlmostEqual(active_color_map.max_iter, ITERATION_MULTIPLIER, places=5)

        # Test with 0x0 dimension value map (engine handles this by returning 1x1 black image)
        self.simulator._value_map.set_size(0,0) # directly manipulate for this test
        self.simulator._width = 0 # to align with generate_bitmap_pil condition
        self.simulator._height = 0
        img_empty = self.simulator.get_current_bitmap_pil_image()
        self.assertEqual(img_empty.size, (1,1))
        self.assertEqual(img_empty.getpixel((0,0)), (0,0,0))
        self.simulator.reset() # Reset to valid state


    def test_randomize_parameters(self):
        test_capacity = 5
        self.simulator.set_capacity(test_capacity)

        # Define which waves will be 'on'
        on_indices = [0, 2] # Example: wave 0 and wave 2 are on
        for i in range(test_capacity):
            self.simulator.wave_infos[i].on = (i in on_indices)
            # Optionally set some initial distinct non-zero, non-random params for 'off' waves
            # to ensure they truly don't change. For now, default (0,0,0) for off waves is fine.

        initial_params = [(wi.on, wi.amplitude, wi.frequency, wi.phase) for wi in self.simulator.wave_infos]

        self.simulator.randomize_parameters()
        self.assertTrue(self.simulator.modified)

        randomized_wave_count = 0
        for i, wi in enumerate(self.simulator.wave_infos):
            current_params = (wi.on, wi.amplitude, wi.frequency, wi.phase)
            if initial_params[i][0]: # If this wave was initially 'on'
                self.assertTrue(wi.on, f"Wave {i} was turned off by randomize_parameters, but should remain on.")
                # Check if its numeric parameters changed
                if (current_params[1:] != initial_params[i][1:]):
                     randomized_wave_count +=1
                # It's possible, though highly unlikely, that random values match initial if they were non-zero.
                # A more robust check for randomization is that values are within expected random range.
                self.assertTrue(MIN_AMPLITUDE <= wi.amplitude <= MAX_AMPLITUDE)
                self.assertTrue(MIN_FREQ_RATIO <= wi.frequency <= MAX_FREQ_RATIO)
                self.assertTrue(MIN_ANGLE <= wi.phase <= MAX_ANGLE)

                # Ensure parameters are not all zero if it was 'on'
                self.assertTrue(wi.amplitude != 0 or wi.frequency != 0 or wi.phase != 0,
                                f"Randomized 'on' wave {i} still has all zero parameters.")
            else: # If this wave was initially 'off'
                self.assertFalse(wi.on, f"Wave {i} was 'off' and should remain 'off'.")
                # Its numeric parameters should NOT have changed
                self.assertEqual(current_params[1:], initial_params[i][1:],
                                 f"Parameters for 'off' wave {i} should not have changed.")

        self.assertEqual(randomized_wave_count, len(on_indices),
                         f"Expected {len(on_indices)} 'on' waves to have their numeric parameters randomized, but {randomized_wave_count} did.")


    def test_file_save_and_load(self):
        test_filepath = os.path.join(self.test_dir, "engine_test_save.chl")

        self.simulator.set_dimensions(60, 80)
        self.simulator.set_capacity(5)
        self.simulator.wave_infos[0] = WaveInfo(on=True, amplitude=0.5, frequency=3.1, phase=45.0)
        self.simulator.wave_infos[1] = WaveInfo(on=False, amplitude=-0.2, frequency=6.0, phase=-90.0)
        # Test boundary values for wave parameters
        self.simulator.wave_infos[2] = WaveInfo(on=True, amplitude=MAX_AMPLITUDE, frequency=MAX_FREQ_RATIO, phase=MAX_ANGLE)
        self.simulator.wave_infos[3] = WaveInfo(on=True, amplitude=MIN_AMPLITUDE, frequency=MIN_FREQ_RATIO, phase=MIN_ANGLE)

        self.simulator.normalize = False
        self.simulator.selected_color_map_name = "Grayscale"
        self.simulator.recalculate_pattern_with_event(None)

        save_success = self.simulator.save_to_file(test_filepath, save_level_map=True)
        self.assertTrue(save_success)
        self.assertFalse(self.simulator.modified)
        self.assertEqual(self.simulator.filename, test_filepath)

        loader_sim = ChladniSimulator()
        load_success = loader_sim.load_from_file(test_filepath)
        self.assertTrue(load_success)

        self.assertEqual(loader_sim.width, 60)
        self.assertEqual(loader_sim.height, 80)
        self.assertEqual(loader_sim.capacity, 5)
        self.assertEqual(len(loader_sim.wave_infos), 5)
        self.assertAlmostEqual(loader_sim.wave_infos[0].amplitude, 0.5)
        self.assertAlmostEqual(loader_sim.wave_infos[1].frequency, 6.0)
        self.assertFalse(loader_sim.normalize)

        # Check boundary values
        self.assertAlmostEqual(loader_sim.wave_infos[2].amplitude, MAX_AMPLITUDE)
        self.assertAlmostEqual(loader_sim.wave_infos[2].frequency, MAX_FREQ_RATIO)
        self.assertAlmostEqual(loader_sim.wave_infos[2].phase, MAX_ANGLE)
        self.assertAlmostEqual(loader_sim.wave_infos[3].amplitude, MIN_AMPLITUDE)
        self.assertAlmostEqual(loader_sim.wave_infos[3].frequency, MIN_FREQ_RATIO)
        self.assertAlmostEqual(loader_sim.wave_infos[3].phase, MIN_ANGLE)


        map_names = list(self.simulator.available_color_maps.keys())
        expected_map_name = "Grayscale"
        if expected_map_name in map_names:
             self.assertEqual(loader_sim.selected_color_map_name, expected_map_name)

        # Test loading a map index that's out of bounds
        # This requires manually creating a .chl file or altering the saved one,
        # which is better suited for test_file_io.py.
        # Here, we trust that the engine's load_from_file correctly handles
        # map_index by defaulting if out of bounds.

        self.assertFalse(loader_sim.modified)
        self.assertEqual(loader_sim.filename, test_filepath)
        self.assertIsNotNone(loader_sim._value_map)
        self.assertFalse(loader_sim._value_map.empty)
        self.assertEqual(loader_sim._value_map.width, 60)
        # _calculated_max_value should be updated from the loaded map
        if loader_sim._value_map._bits is not None and loader_sim._value_map._bits.size > 0:
            expected_max_val = np.max(loader_sim._value_map._bits)
            self.assertAlmostEqual(loader_sim._calculated_max_value, expected_max_val)
        else:
            self.assertEqual(loader_sim._calculated_max_value, 0)


    def test_file_save_no_level_map(self):
        test_filepath = os.path.join(self.test_dir, "engine_test_no_map.chl")
        # Ensure a valid frequency to get a non-zero pattern
        self.simulator.wave_infos[0] = WaveInfo(on=True, amplitude=0.7, frequency=2.0)
        self.simulator.recalculate_pattern_with_event(None) # calculate something first
        original_max_val = self.simulator._calculated_max_value
        self.assertGreater(original_max_val, 0) # Ensure there was something

        save_success = self.simulator.save_to_file(test_filepath, save_level_map=False)
        self.assertTrue(save_success)

        loader_sim = ChladniSimulator()
        load_success = loader_sim.load_from_file(test_filepath)
        self.assertTrue(load_success)

        self.assertEqual(loader_sim.width, self.simulator.width)
        self.assertEqual(loader_sim.height, self.simulator.height)
        self.assertEqual(loader_sim._value_map.width, self.simulator.width) # ValueMap should be sized
        self.assertEqual(loader_sim._value_map.height, self.simulator.height)
        # If no level map, _bits should be all zeros, and _calculated_max_value should be 0
        if loader_sim._value_map._bits is not None:
             self.assertTrue(np.all(loader_sim._value_map._bits == 0))
        self.assertEqual(loader_sim._calculated_max_value, 0)

    def test_save_load_max_capacity(self):
        # This test might be slow if max capacity is very large.
        # The practical max capacity is not defined, but set_capacity handles list extension.
        # Let's test with a reasonably large capacity.
        capacity = 100 # Example large capacity
        self.simulator.set_capacity(capacity)
        for i in range(capacity):
            self.simulator.wave_infos[i] = WaveInfo(True, 0.1 * (i % 10), 1.0 + i*0.1, float(i*10 % 360))

        self.simulator.recalculate_pattern_with_event(None)
        test_filepath = os.path.join(self.test_dir, "engine_max_cap.chl")
        save_success = self.simulator.save_to_file(test_filepath, save_level_map=True)
        self.assertTrue(save_success)

        loader_sim = ChladniSimulator()
        load_success = loader_sim.load_from_file(test_filepath)
        self.assertTrue(load_success)
        self.assertEqual(loader_sim.capacity, capacity)
        self.assertEqual(len(loader_sim.wave_infos), capacity)
        self.assertAlmostEqual(loader_sim.wave_infos[capacity-1].frequency, 1.0 + (capacity-1)*0.1, delta=1e-6)

    def test_load_from_file_error_handling(self):
        # Test loading a non-existent file
        non_existent_filepath = os.path.join(self.test_dir, "does_not_exist.chl")
        load_success = self.simulator.load_from_file(non_existent_filepath)
        self.assertFalse(load_success)
        # State should ideally remain unchanged or be reset.
        # Current ChladniSimulator.load_from_file prints error and returns False, doesn't reset.
        # This is acceptable, but good to be aware of.

        # Test loading a corrupted file (basic check, more detailed in test_file_io.py)
        corrupted_filepath = os.path.join(self.test_dir, "corrupted.chl")
        with open(corrupted_filepath, "wb") as f:
            f.write(b"this is not a valid chl file")
        load_success = self.simulator.load_from_file(corrupted_filepath)
        self.assertFalse(load_success)

    def test_get_current_color_map(self):
        self.simulator.selected_color_map_name = "Grayscale"
        cm_gray = self.simulator.get_current_color_map()
        self.assertEqual(cm_gray.name, "Grayscale")

        self.simulator.selected_color_map_name = "Spectrum"
        cm_spectrum = self.simulator.get_current_color_map()
        self.assertEqual(cm_spectrum.name, "Spectrum")

        # Test fallback for non-existent map name
        self.simulator.selected_color_map_name = "NonExistentMap"
        cm_fallback = self.simulator.get_current_color_map()
        # It should fallback to "Grayscale" as per current implementation
        self.assertEqual(cm_fallback.name, "Grayscale")
        self.assertEqual(cm_fallback, DEFAULT_COLOR_MAPS["Grayscale"])


if __name__ == '__main__':
    unittest.main()
