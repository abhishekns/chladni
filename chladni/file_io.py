import gzip
import struct
from typing import NamedTuple, BinaryIO

from .core import WaveInfo, ValueMap, MIN_AMPLITUDE, MAX_AMPLITUDE, MIN_FREQ_RATIO, MAX_FREQ_RATIO, MIN_ANGLE, MAX_ANGLE

# Constants from ucladni.pas
CHL_ID = b'CHL\x00' # Packed array[0..3] of UTF8Char, original was #164'CHL'. The last char might be null or space.
                   # Let's assume it's CHL + a padding/null byte if it's 4 bytes.
                   # The Pascal code uses #164'CHL', which means the byte value 164 followed by 'C', 'H', 'L'.
                   # However, typical Delphi/Lazarus packed array of char for IDs is often just the chars.
                   # Let's re-evaluate CHL_ID. If it's `packed array[0..3] of UTF8Char` initialized with `'CHL'`,
                   # it would be 'C', 'H', 'L', '\0'.
CHL_ID_EXPECTED = b'\xA4CHL' # Standard interpretation of a 4-char array initialized with #164"CHL"
CHL_UNTITLED = 'Untitled.chl'

# File format versions
# Version 10: Standard save
# Version 11: Includes fSaveLevelMapWithFile data
VERSION_STANDARD = 10
VERSION_WITH_LEVEL_MAP = 11

# Helper to ensure range, similar to Pascal's EnsureRange
def ensure_range(value, min_val, max_val):
    return max(min_val, min(value, max_val))

class ChladniData(NamedTuple):
    wave_infos: list[WaveInfo]
    map_index: int
    width: int
    height: int
    normalize: bool
    value_map: ValueMap | None # Only if loaded/saved with level map data
    filename: str = CHL_UNTITLED # For context, not part of the file itself but useful metadata

def _read_wave_info(f: BinaryIO) -> WaveInfo:
    # TWaveInfo = record
    #   &On: boolean;        // Pascal Boolean is often 1 byte, 0 or 1
    #   Amplitude: single;   // 4 bytes
    #   Frequency: single;   // 4 bytes
    #   Phase: single;       // 4 bytes
    # end;
    # Total size: 1 + 4 + 4 + 4 = 13 bytes
    on_byte = struct.unpack('?', f.read(1))[0] # '?' is for bool (1 byte)
    amplitude = struct.unpack('f', f.read(4))[0]
    frequency = struct.unpack('f', f.read(4))[0]
    phase = struct.unpack('f', f.read(4))[0]
    return WaveInfo(
        on=on_byte,
        amplitude=amplitude,
        frequency=frequency,
        phase=phase
    )

def _write_wave_info(f: BinaryIO, wi: WaveInfo):
    f.write(struct.pack('?', wi.on))
    f.write(struct.pack('f', wi.amplitude))
    f.write(struct.pack('f', wi.frequency))
    f.write(struct.pack('f', wi.phase))

def load_chl_file(filepath: str) -> ChladniData:
    """Loads a Chladni (.chl) file."""
    with gzip.open(filepath, 'rb') as gz_f:
        # Read ID and Version
        file_id = gz_f.read(4)
        if file_id != CHL_ID_EXPECTED:
            # The original code has a specific error message check: #164'CHL'
            # Let's try to match that if the simple CHL\0 fails.
            # 164 is 0xA4. So it could be b'\xA4CHL'.
            # For now, stick to one interpretation or make it flexible if tests fail.
            raise ValueError(f"Invalid CHL file ID. Expected {CHL_ID_EXPECTED!r}, got {file_id!r}")

        version = struct.unpack('<H', gz_f.read(2))[0] # '<H' for unsigned short (2 bytes), little-endian

        if not (version == VERSION_STANDARD or version == VERSION_WITH_LEVEL_MAP):
            raise ValueError(f"Unsupported CHL file version. Expected {VERSION_STANDARD} or {VERSION_WITH_LEVEL_MAP}, got {version}")

        save_level_map_with_file = (version == VERSION_WITH_LEVEL_MAP)

        # Read data
        capacity = struct.unpack('<I', gz_f.read(4))[0] # '<I' for unsigned int (4 bytes)
        map_index = struct.unpack('<I', gz_f.read(4))[0]
        width = struct.unpack('<I', gz_f.read(4))[0]
        height = struct.unpack('<I', gz_f.read(4))[0]
        normalize_byte = struct.unpack('B', gz_f.read(1))[0] # 'B' for unsigned char (1 byte)
        normalize = (normalize_byte > 0)

        wave_infos = []
        for _ in range(capacity):
            wave_infos.append(_read_wave_info(gz_f))

        value_map_data = None
        if save_level_map_with_file:
            value_map_data = ValueMap(width, height)
            if width > 0 and height > 0 : # Prevent reading if size is 0, though unlikely for saved maps
                expected_bytes = width * height * 4 # 4 bytes per float32
                raw_floats_data = gz_f.read(expected_bytes)
                if len(raw_floats_data) != expected_bytes:
                    raise IOError(f"Could not read expected level map data. Expected {expected_bytes}, got {len(raw_floats_data)}")

                # Pascal stores single as 4-byte float. NumPy default float is float64 (8 bytes).
                # Need to ensure we load as float32.
                floats = np.frombuffer(raw_floats_data, dtype=np.float32)
                value_map_data._bits = floats.reshape((height, width))
            elif width == 0 or height == 0: # if saved with zero dimensions
                 value_map_data._bits = np.array([], dtype=np.float32).reshape((height,width))


        return ChladniData(
            wave_infos=wave_infos,
            map_index=map_index,
            width=width,
            height=height,
            normalize=normalize,
            value_map=value_map_data,
            filename=filepath
        )

def save_chl_file(filepath: str, data: ChladniData, save_level_map: bool = False):
    """Saves Chladni data to a .chl file."""
    version = VERSION_WITH_LEVEL_MAP if save_level_map and data.value_map is not None else VERSION_STANDARD

    with gzip.open(filepath, 'wb') as gz_f:
        gz_f.write(CHL_ID_EXPECTED)
        gz_f.write(struct.pack('<H', version)) # Little-endian unsigned short

        gz_f.write(struct.pack('<I', len(data.wave_infos))) # Capacity
        gz_f.write(struct.pack('<I', data.map_index))
        gz_f.write(struct.pack('<I', data.width))
        gz_f.write(struct.pack('<I', data.height))
        gz_f.write(struct.pack('B', 1 if data.normalize else 0)) # Boolean as 1 byte

        for wi in data.wave_infos:
            _write_wave_info(gz_f, wi)

        if version == VERSION_WITH_LEVEL_MAP and data.value_map is not None and data.value_map._bits is not None:
            # Ensure the ValueMap's internal _bits array is C-contiguous and float32
            if not data.value_map._bits.flags['C_CONTIGUOUS']:
                contiguous_bits = np.ascontiguousarray(data.value_map._bits, dtype=np.float32)
            else:
                contiguous_bits = data.value_map._bits.astype(np.float32, copy=False) # Ensure float32 without unnecessary copy

            gz_f.write(contiguous_bits.tobytes())

# Need to import numpy for frombuffer and array operations in load_chl_file
import numpy as np

if __name__ == '__main__':
    print("Chladni File I/O Basic Test")

    # Create dummy data
    test_waves = [
        WaveInfo(on=True, amplitude=0.5, frequency=3.0, phase=90.0),
        WaveInfo(on=False, amplitude=-0.2, frequency=5.5, phase=-45.0),
        WaveInfo(on=True, amplitude=1.0, frequency=7.0, phase=180.0),
    ]
    test_width, test_height = 10, 8 # Small dimensions for test

    # Create a dummy ValueMap for saving
    vm_to_save = ValueMap(test_width, test_height)
    if vm_to_save._bits is not None: # Should not be None after init with size > 0
        for r in range(test_height):
            for c in range(test_width):
                vm_to_save._bits[r, c] = float(r * test_width + c)

    chl_data_to_save = ChladniData(
        wave_infos=test_waves,
        map_index=5,
        width=test_width,
        height=test_height,
        normalize=True,
        value_map=vm_to_save
    )

    test_filepath_map = "test_save_with_map.chl"
    test_filepath_no_map = "test_save_no_map.chl"

    try:
        # Test saving with level map
        print(f"Saving to {test_filepath_map} (with level map)...")
        save_chl_file(test_filepath_map, chl_data_to_save, save_level_map=True)
        print("Save successful.")

        # Test loading with level map
        print(f"Loading from {test_filepath_map}...")
        loaded_data_map = load_chl_file(test_filepath_map)
        print("Load successful.")
        assert loaded_data_map.width == test_width
        assert loaded_data_map.height == test_height
        assert loaded_data_map.normalize == True
        assert len(loaded_data_map.wave_infos) == len(test_waves)
        assert loaded_data_map.wave_infos[0].amplitude == test_waves[0].amplitude
        assert loaded_data_map.value_map is not None
        if loaded_data_map.value_map._bits is not None and vm_to_save._bits is not None:
             assert np.array_equal(loaded_data_map.value_map._bits, vm_to_save._bits)
        print("Loaded data (with map) matches saved data.")

        # Test saving without level map
        print(f"Saving to {test_filepath_no_map} (without level map)...")
        save_chl_file(test_filepath_no_map, chl_data_to_save, save_level_map=False)
        print("Save successful.")

        # Test loading without level map
        print(f"Loading from {test_filepath_no_map}...")
        loaded_data_no_map = load_chl_file(test_filepath_no_map)
        print("Load successful.")
        assert loaded_data_no_map.value_map is None
        assert loaded_data_no_map.width == test_width # Other fields should still be correct
        print("Loaded data (without map) has no level map as expected.")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Clean up test files
        import os
        if os.path.exists(test_filepath_map):
            os.remove(test_filepath_map)
        if os.path.exists(test_filepath_no_map):
            os.remove(test_filepath_no_map)
        print("Test files cleaned up.")

    print("File I/O tests completed.")
