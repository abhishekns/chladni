import unittest
import os
import shutil
import gzip
import struct
import numpy as np

# Adjust import path for tests
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.file_io import load_chl_file, save_chl_file, ChladniData, CHL_ID_EXPECTED, VERSION_STANDARD, VERSION_WITH_LEVEL_MAP
from src.core import WaveInfo, ValueMap


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

if __name__ == '__main__':
    unittest.main()
