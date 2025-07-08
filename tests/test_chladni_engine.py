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

from src.chladni_engine import ChladniSimulator, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_CAPACITY, CHL_UNTITLED, WaveInfo
from src.core import ValueMap


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

    def test_set_dimensions(self):
        self.simulator.set_dimensions(100, 150)
        self.assertEqual(self.simulator.width, 100)
        self.assertEqual(self.simulator.height, 150)
        self.assertEqual(self.simulator._value_map.width, 100)
        self.assertEqual(self.simulator._value_map.height, 150)
        self.assertTrue(self.simulator.modified)

    def test_set_capacity(self):
        self.simulator.set_capacity(50)
        self.assertEqual(self.simulator.capacity, 50)
        self.assertEqual(len(self.simulator.wave_infos), 50)
        self.assertTrue(self.simulator.modified)

        self.simulator.set_capacity(20)
        self.assertEqual(self.simulator.capacity, 20)
        self.assertEqual(len(self.simulator.wave_infos), 20)

    def test_reset(self):
        self.simulator.set_dimensions(100, 100)
        self.simulator.set_capacity(10)
        self.simulator.filename = "test.chl"
        self.simulator.modified = True
        if self.simulator.wave_infos: # Ensure list is not empty
            self.simulator.wave_infos[0].amplitude = 1.0

        self.simulator.reset()

        # Check against default values after reset
        self.assertEqual(self.simulator.width, DEFAULT_WIDTH)
        self.assertEqual(self.simulator.height, DEFAULT_HEIGHT)
        self.assertEqual(self.simulator.capacity, DEFAULT_CAPACITY)
        self.assertEqual(self.simulator.filename, CHL_UNTITLED)
        self.assertFalse(self.simulator.modified)
        if self.simulator.wave_infos:
             self.assertEqual(self.simulator.wave_infos[0].amplitude, 0.0)


    def test_recalculate_pattern_and_get_image(self):
        if self.simulator.wave_infos:
            self.simulator.wave_infos[0] = WaveInfo(on=True, amplitude=1.0, frequency=4.0, phase=0.0)

        self.simulator.recalculate_pattern_with_event(None) # Use the new method
        self.assertTrue(self.simulator.modified)
        self.assertGreater(self.simulator._calculated_max_value, 0)

        img = self.simulator.get_current_bitmap_pil_image()
        self.assertIsInstance(img, Image.Image)
        self.assertEqual(img.size, (self.simulator.width, self.simulator.height))

    def test_randomize_parameters(self):
        # Ensure enough waves for the test, and set specific 'on' states
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
                # For simplicity, we assume non-zero initial 'on' waves will change.
                # If initial 'on' wave params were 0,0,0, this check might be tricky.
                # Let's assume they are not 0,0,0 if 'on' for this test or that random will make them non-zero.
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
        self.simulator.normalize = False
        self.simulator.selected_color_map_name = "Grayscale"
        self.simulator.recalculate_pattern_with_event(None) # Use the new method

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
        self.assertEqual(loader_sim.wave_infos[0].amplitude, 0.5)
        self.assertEqual(loader_sim.wave_infos[1].frequency, 6.0)
        self.assertFalse(loader_sim.normalize)

        map_names = list(self.simulator.available_color_maps.keys())
        expected_map_name = "Grayscale"
        if expected_map_name in map_names:
             self.assertEqual(loader_sim.selected_color_map_name, expected_map_name)

        self.assertFalse(loader_sim.modified)
        self.assertEqual(loader_sim.filename, test_filepath)
        self.assertIsNotNone(loader_sim._value_map)
        self.assertFalse(loader_sim._value_map.empty)
        self.assertEqual(loader_sim._value_map.width, 60)
        self.assertGreater(loader_sim._calculated_max_value, 0)


    def test_file_save_no_level_map(self):
        test_filepath = os.path.join(self.test_dir, "engine_test_no_map.chl")
        self.simulator.wave_infos[0] = WaveInfo(on=True, amplitude=0.7)

        save_success = self.simulator.save_to_file(test_filepath, save_level_map=False)
        self.assertTrue(save_success)

        loader_sim = ChladniSimulator()
        load_success = loader_sim.load_from_file(test_filepath)
        self.assertTrue(load_success)

        self.assertEqual(loader_sim.width, self.simulator.width)
        self.assertEqual(loader_sim.height, self.simulator.height)
        self.assertEqual(loader_sim._value_map.width, self.simulator.width)
        self.assertEqual(loader_sim._value_map.height, self.simulator.height)
        if loader_sim._value_map._bits is not None:
             self.assertTrue(np.all(loader_sim._value_map._bits == 0))
        self.assertEqual(loader_sim._calculated_max_value, 0)

if __name__ == '__main__':
    unittest.main()
