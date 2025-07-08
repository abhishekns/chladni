import unittest
import os
import shutil
import gzip
import struct
import numpy as np

# Adjust import path for tests
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chladni.file_io import load_chl_file, save_chl_file, ChladniData, CHL_ID_EXPECTED, VERSION_STANDARD, VERSION_WITH_LEVEL_MAP
from chladni.core import WaveInfo, ValueMap


class TestFileIO(unittest.TestCase):

    def setUp(self):
        self.test_dir = "temp_file_io_test_files" # Unique name
        os.makedirs(self.test_dir, exist_ok=True)

        self.test_waves = [
            WaveInfo(on=True, amplitude=0.5, frequency=3.0, phase=90.0),
            WaveInfo(on=False, amplitude=-0.2, frequency=5.5, phase=-45.0),
            WaveInfo(on=True, amplitude=1.0, frequency=7.0, phase=180.0),
        ]
        self.test_width, self.test_height = 10, 8
        self.vm_to_save = ValueMap(self.test_width, self.test_height)
        if self.vm_to_save._bits is not None: # Should be initialized by ValueMap constructor
            for r_idx in range(self.test_height):
                for c_idx in range(self.test_width):
                    # Ensure we are assigning to an existing array
                    self.vm_to_save._bits[r_idx, c_idx] = float(r_idx * self.test_width + c_idx)

        self.chl_data_to_save = ChladniData(
            wave_infos=self.test_waves,
            map_index=5,
            width=self.test_width,
            height=self.test_height,
            normalize=True,
            value_map=self.vm_to_save,
            filename="dummy.chl"
        )

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_save_and_load_with_level_map(self):
        test_filepath = os.path.join(self.test_dir, "test_with_map.chl")

        save_chl_file(test_filepath, self.chl_data_to_save, save_level_map=True)
        self.assertTrue(os.path.exists(test_filepath))

        loaded_data = load_chl_file(test_filepath)

        self.assertEqual(loaded_data.width, self.test_width)
        self.assertEqual(loaded_data.height, self.test_height)
        self.assertTrue(loaded_data.normalize)
        self.assertEqual(len(loaded_data.wave_infos), len(self.test_waves))
        self.assertEqual(loaded_data.wave_infos[0].amplitude, self.test_waves[0].amplitude)
        self.assertIsNotNone(loaded_data.value_map)
        self.assertIsNotNone(loaded_data.value_map._bits) # Ensure _bits is not None
        self.assertTrue(np.array_equal(loaded_data.value_map._bits, self.vm_to_save._bits))

    def test_save_and_load_without_level_map(self):
        test_filepath = os.path.join(self.test_dir, "test_no_map.chl")

        save_chl_file(test_filepath, self.chl_data_to_save, save_level_map=False)
        self.assertTrue(os.path.exists(test_filepath))

        loaded_data = load_chl_file(test_filepath)

        self.assertIsNone(loaded_data.value_map) # ValueMap object itself should be None
        self.assertEqual(loaded_data.width, self.test_width)
        self.assertEqual(loaded_data.wave_infos[2].phase, self.test_waves[2].phase)

    def test_load_invalid_file_id(self):
        test_filepath = os.path.join(self.test_dir, "bad_id.chl")
        with gzip.open(test_filepath, 'wb') as gz_f:
            gz_f.write(b"BAD!")
            gz_f.write(struct.pack('<H', VERSION_STANDARD))

        with self.assertRaisesRegex(ValueError, "Invalid CHL file ID"):
            load_chl_file(test_filepath)

    def test_load_unsupported_version(self):
        test_filepath = os.path.join(self.test_dir, "bad_version.chl")
        with gzip.open(test_filepath, 'wb') as gz_f:
            gz_f.write(CHL_ID_EXPECTED)
            gz_f.write(struct.pack('<H', 99)) # Unsupported version

        with self.assertRaisesRegex(ValueError, "Unsupported CHL file version"):
            load_chl_file(test_filepath)

    def test_load_corrupted_level_map_data(self):
        test_filepath = os.path.join(self.test_dir, "corrupted_map.chl")

        with gzip.open(test_filepath, 'wb') as gz_f:
            gz_f.write(CHL_ID_EXPECTED)
            gz_f.write(struct.pack('<H', VERSION_WITH_LEVEL_MAP))
            gz_f.write(struct.pack('<I', len(self.chl_data_to_save.wave_infos)))
            gz_f.write(struct.pack('<I', self.chl_data_to_save.map_index))
            gz_f.write(struct.pack('<I', self.chl_data_to_save.width))
            gz_f.write(struct.pack('<I', self.chl_data_to_save.height))
            gz_f.write(struct.pack('B', 1 if self.chl_data_to_save.normalize else 0))
            for wi in self.chl_data_to_save.wave_infos:
                gz_f.write(struct.pack('?', wi.on))
                gz_f.write(struct.pack('f', wi.amplitude))
                gz_f.write(struct.pack('f', wi.frequency))
                gz_f.write(struct.pack('f', wi.phase))

            if self.chl_data_to_save.value_map and self.chl_data_to_save.value_map._bits is not None:
                full_map_bytes = self.chl_data_to_save.value_map._bits.astype(np.float32).tobytes()
                gz_f.write(full_map_bytes[:len(full_map_bytes)//2]) # Half the data

        with self.assertRaisesRegex(IOError, "Could not read expected level map data"):
            load_chl_file(test_filepath)

    def test_load_malformed_data_various(self):
        # Test case 1: Incorrect number of wave info items (too few bytes for last wave)
        test_filepath_wave_count = os.path.join(self.test_dir, "bad_wave_count.chl")
        with gzip.open(test_filepath_wave_count, 'wb') as gz_f:
            gz_f.write(CHL_ID_EXPECTED)
            gz_f.write(struct.pack('<H', VERSION_STANDARD))
            gz_f.write(struct.pack('<I', 3)) # Claim 3 waves
            gz_f.write(struct.pack('<I', 0)) # map_index
            gz_f.write(struct.pack('<I', 10)) # width
            gz_f.write(struct.pack('<I', 10)) # height
            gz_f.write(struct.pack('B', 1))   # normalize
            # Write 2 full waves
            for wi in self.test_waves[:2]:
                gz_f.write(struct.pack('?', wi.on))
                gz_f.write(struct.pack('f', wi.amplitude))
                gz_f.write(struct.pack('f', wi.frequency))
                gz_f.write(struct.pack('f', wi.phase))
            # Write partial 3rd wave (e.g., only 'on' and 'amplitude')
            gz_f.write(struct.pack('?', self.test_waves[2].on))
            gz_f.write(struct.pack('f', self.test_waves[2].amplitude))
            # Missing frequency and phase for the 3rd wave

        with self.assertRaises((struct.error, EOFError)): # struct.error if not enough bytes for unpack
            load_chl_file(test_filepath_wave_count)

        # Test case 2: Negative width/height (struct.pack will handle unsigned, so load will read as large positive)
        # This scenario is more about how the ChladniSimulator handles these values later.
        # file_io will load them as large unsigned integers. No error expected here from file_io itself.

        # Test case 3: Data truncation at various points
        base_data = bytearray()
        base_data.extend(CHL_ID_EXPECTED)
        base_data.extend(struct.pack('<H', VERSION_STANDARD))
        base_data.extend(struct.pack('<I', 1)) # capacity
        base_data.extend(struct.pack('<I', 0)) # map_index
        base_data.extend(struct.pack('<I', 10)) # width
        # Truncate before height
        truncated_filepath_1 = os.path.join(self.test_dir, "truncated1.chl")
        with gzip.open(truncated_filepath_1, 'wb') as gz_f:
            gz_f.write(base_data)
        with self.assertRaises((struct.error, EOFError)):
            load_chl_file(truncated_filepath_1)

        base_data.extend(struct.pack('<I', 10)) # height
        base_data.extend(struct.pack('B', 1))   # normalize
        # Truncate before wave data
        truncated_filepath_2 = os.path.join(self.test_dir, "truncated2.chl")
        with gzip.open(truncated_filepath_2, 'wb') as gz_f:
            gz_f.write(base_data)
        with self.assertRaises((struct.error, EOFError)):
            load_chl_file(truncated_filepath_2)


    def test_save_empty_wave_infos(self):
        test_filepath = os.path.join(self.test_dir, "empty_waves.chl")
        data_empty_waves = ChladniData(
            wave_infos=[],
            map_index=0,
            width=10,
            height=10,
            normalize=False,
            value_map=None
        )
        save_chl_file(test_filepath, data_empty_waves, save_level_map=False)
        self.assertTrue(os.path.exists(test_filepath))

        loaded_data = load_chl_file(test_filepath)
        self.assertEqual(len(loaded_data.wave_infos), 0)
        self.assertEqual(loaded_data.width, 10)

    def test_save_value_map_zero_dimensions(self):
        test_filepath = os.path.join(self.test_dir, "zero_dim_map.chl")
        vm_zero = ValueMap(0, 0)
        data_zero_map = ChladniData(
            wave_infos=self.test_waves[:1],
            map_index=0,
            width=0,
            height=0,
            normalize=False,
            value_map=vm_zero
        )
        # Try saving with level map True, even if dimensions are zero
        save_chl_file(test_filepath, data_zero_map, save_level_map=True)
        self.assertTrue(os.path.exists(test_filepath))

        loaded_data = load_chl_file(test_filepath)
        self.assertEqual(loaded_data.width, 0)
        self.assertEqual(loaded_data.height, 0)
        self.assertIsNotNone(loaded_data.value_map) # ValueMap object should be created
        self.assertTrue(loaded_data.value_map.empty) # And it should be empty
        if loaded_data.value_map._bits is not None: # Check internal _bits structure
             self.assertEqual(loaded_data.value_map._bits.size, 0)


if __name__ == '__main__':
    unittest.main()
