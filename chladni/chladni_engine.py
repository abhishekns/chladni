from typing import List, Tuple
from PIL import Image

import threading # For stop_event
from .core import ValueMap, WaveInfo, calculate_chladni_pattern, ITERATION_MULTIPLIER, MIN_FREQ_RATIO, MAX_FREQ_RATIO, MIN_AMPLITUDE, MAX_AMPLITUDE, MIN_ANGLE, MAX_ANGLE
from .visualization import ColorMap, DEFAULT_COLOR_MAPS, generate_bitmap_pil
from .file_io import load_chl_file, save_chl_file, ChladniData, CHL_UNTITLED

# For random parameter generation
import random

DEFAULT_WIDTH = 500
DEFAULT_HEIGHT = 500
DEFAULT_CAPACITY = 10
DEFAULT_NORMALIZE = True
DEFAULT_MAP_NAME = "Spectrum" # Default colormap name

class ChladniSimulator:
    def __init__(self):
        self.filename: str = CHL_UNTITLED
        self.modified: bool = False

        self._width: int = DEFAULT_WIDTH
        self._height: int = DEFAULT_HEIGHT
        self._capacity: int = DEFAULT_CAPACITY
        self.normalize: bool = DEFAULT_NORMALIZE

        self.wave_infos: List[WaveInfo] = []
        self._value_map: ValueMap = ValueMap(self._width, self._height)
        self._calculated_max_value: float = 0.0 # Stores the max value from the last calculation

        self.available_color_maps = DEFAULT_COLOR_MAPS
        self.selected_color_map_name: str = DEFAULT_MAP_NAME

        self.reset() # Initialize with default state

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def capacity(self) -> int:
        return self._capacity

    def set_dimensions(self, width: int, height: int):
        width = max(10, min(width, 16384)) # Same range as Pascal TCladni.SetSize
        height = max(10, min(height, 16384))
        if self._width != width or self._height != height:
            self._width = width
            self._height = height
            self._value_map.set_size(self._width, self._height)
            # Dimensions change implies recalculation is needed, but don't set modified yet
            # Or, set modified if a pattern was already there. For now, simple.
            self.modified = True

    def set_capacity(self, capacity: int):
        capacity = max(1, capacity) # Must have at least 1
        if self._capacity != capacity:
            self._capacity = capacity
            # Adjust wave_infos list size
            current_len = len(self.wave_infos)
            if current_len < self._capacity:
                for _ in range(self._capacity - current_len):
                    self.wave_infos.append(WaveInfo())
            elif current_len > self._capacity:
                self.wave_infos = self.wave_infos[:self._capacity]
            self.modified = True

    def reset(self):
        self._width = DEFAULT_WIDTH
        self._height = DEFAULT_HEIGHT
        self._capacity = DEFAULT_CAPACITY # Call set_capacity to ensure wave_infos is also reset
        self.normalize = DEFAULT_NORMALIZE
        self.selected_color_map_name = DEFAULT_MAP_NAME

        self._value_map.set_size(self._width, self._height)
        self._value_map.clear()
        self._calculated_max_value = 0.0

        self.wave_infos = [WaveInfo() for _ in range(self._capacity)]
        # Example: Turn on the first wave for a default pattern (optional)
        # if self._capacity > 0:
        #     self.wave_infos[0] = WaveInfo(on=True, amplitude=0.5, frequency=5, phase=0)

        self.filename = CHL_UNTITLED
        self.modified = False # Resetting to a clean state

    def get_current_color_map(self) -> ColorMap:
        return self.available_color_maps.get(self.selected_color_map_name, self.available_color_maps["Grayscale"])

    def recalculate_pattern(self):
        """Calculates the Chladni pattern based on current wave_infos and dimensions."""
        if self._width == 0 or self._height == 0:
            return # Or raise error

        self._calculated_max_value = calculate_chladni_pattern(
            self._value_map,
            self.wave_infos,
            self._width,
            self._height
        )
        self.modified = True # Calculation implies data has changed or been generated
        # print(f"Engine: Recalculated pattern. Max value: {self._calculated_max_value}") # For debugging

    def recalculate_pattern_with_event(self, stop_event: threading.Event | None): # Renamed for clarity
        """Calculates the Chladni pattern, supporting a stop event."""
        if self._width == 0 or self._height == 0:
            return

        self._calculated_max_value = calculate_chladni_pattern(
            self._value_map,
            self.wave_infos,
            self._width,
            self.height, # Corrected from self._height to self.height (though they are same via property)
            stop_event=stop_event
        )
        # Only mark as modified if not stopped early, or always?
        # If stopped early, the pattern is incomplete.
        # Let's assume any calculation attempt, even if stopped, might change things.
        if not (stop_event and stop_event.is_set()):
            self.modified = True


    def get_current_bitmap_pil_image(self, recalculate_if_needed: bool = False) -> Image.Image:
        """
        Generates a PIL image from the current value_map and color_map settings.
        If recalculate_if_needed is True, it will first recalculate the pattern.
        (Note: For now, this method might be simpler if it assumes calculation is done,
         or if the GUI explicitly calls recalculate then this.)
        Let's assume for now that recalculation is handled separately by the GUI trigger.
        """
        # For GUI, it's often better to have explicit recalculate and then get_image calls.
        # This method will just render the current state of _value_map.

        color_map_to_use = self.get_current_color_map()

        # The generate_bitmap_pil function uses current_max_value_in_map if normalize is True.
        # This current_max_value_in_map should be self._calculated_max_value from the last calculation.
        return generate_bitmap_pil(
            self._value_map,
            color_map_to_use,
            current_max_value_in_map=self._calculated_max_value,
            normalize=self.normalize
        )

    def randomize_parameters(self):
        for i in range(len(self.wave_infos)):
            if self.wave_infos[i].on : # Only randomize active waves, or all? Original randomizes if on.
                                     # Let's assume we randomize only if 'on' is true.
                                     # Or, more simply, always randomize but user can toggle 'on'
                self.wave_infos[i].amplitude = random.uniform(MIN_AMPLITUDE, MAX_AMPLITUDE)
                self.wave_infos[i].frequency = random.uniform(MIN_FREQ_RATIO, MAX_FREQ_RATIO)
                self.wave_infos[i].phase = random.uniform(MIN_ANGLE, MAX_ANGLE)
        self.modified = True
        # Caller should then trigger recalculate_pattern()

    def load_from_file(self, filepath: str) -> bool:
        try:
            data = load_chl_file(filepath)

            self._width = data.width
            self._height = data.height
            self.set_capacity(len(data.wave_infos)) # This also updates self.wave_infos list size
            self.wave_infos = data.wave_infos # Assign loaded wave_infos

            self.normalize = data.normalize
            # Find map_index. The original chl stores map_index. We store map_name.
            # For now, let's ignore map_index from file or try a simple mapping if possible.
            # The default listbox in GUI will select by index, so we need to provide that.
            # This needs more thought on how to map saved index to our named maps.
            # For now, set to default or try to find by a common name if index is 0, 1 etc.
            map_names = list(self.available_color_maps.keys())
            if 0 <= data.map_index < len(map_names):
                 self.selected_color_map_name = map_names[data.map_index]
            else:
                self.selected_color_map_name = DEFAULT_MAP_NAME


            if data.value_map and data.value_map._bits is not None:
                self._value_map = data.value_map
                # If value_map is loaded, we need to know its max_value if we want to normalize correctly
                # without recalculating. The .chl file doesn't store fValMax explicitly.
                # So, if a level map is loaded, we might need to calculate its max for normalization,
                # or the user must recalculate.
                # For now, if map is loaded, subsequent get_current_bitmap will use it.
                # The _calculated_max_value should ideally be derived from this loaded map if used directly.
                # This is tricky. Simplest is: if loaded, it's there. If user hits render, it's new.
                # Let's assume if level map is loaded, it's used as is. Normalization might be off
                # until next recalculation if fValMax is not part of .chl
                self._calculated_max_value = np.max(self._value_map._bits) if self._value_map._bits is not None and self._value_map._bits.size > 0 else 0

            else:
                # No value map in file, so it needs calculation.
                self._value_map.set_size(self._width, self._height)
                self._value_map.clear()
                self._calculated_max_value = 0 # Needs recalculation
                # self.recalculate_pattern() # Or let GUI decide to call recalculate

            self.filename = filepath
            self.modified = False # Freshly loaded
            return True
        except Exception as e:
            print(f"Error loading CHL file in engine: {e}")
            # Potentially reset to a default state on error
            # self.reset()
            return False

    def save_to_file(self, filepath: str, save_level_map: bool = True) -> bool:
        # Ensure value_map has correct dimensions if it was never calculated but we save it
        if self._value_map.width != self._width or self._value_map.height != self._height:
            self._value_map.set_size(self._width, self._height)
            # If we save level map, it should ideally be after a calculation.
            # If it's empty and we save it, that's fine.

        # Map selected_color_map_name back to an index for saving
        map_index = 0
        map_names = list(self.available_color_maps.keys())
        try:
            map_index = map_names.index(self.selected_color_map_name)
        except ValueError:
            map_index = 0 # Default to first map if current selection not found (should not happen)

        data_to_save = ChladniData(
            wave_infos=self.wave_infos,
            map_index=map_index,
            width=self._width,
            height=self._height,
            normalize=self.normalize,
            value_map=self._value_map if save_level_map else None,
            filename=filepath
        )
        try:
            save_chl_file(filepath, data_to_save, save_level_map=save_level_map)
            self.filename = filepath
            self.modified = False # Saved state is now clean
            return True
        except Exception as e:
            print(f"Error saving CHL file in engine: {e}")
            return False

if __name__ == '__main__':
    # Basic test for ChladniSimulator
    engine = ChladniSimulator()
    print(f"Initial state: {engine.width}x{engine.height}, {engine.capacity} waves, File: {engine.filename}")

    # Modify some parameters
    engine.set_dimensions(100, 100)
    engine.set_capacity(10)
    if engine.wave_infos:
        engine.wave_infos[0] = WaveInfo(on=True, amplitude=1.0, frequency=5, phase=0)
        engine.wave_infos[1] = WaveInfo(on=True, amplitude=0.7, frequency=3, phase=90)

    engine.recalculate_pattern()
    print(f"After recalculation: Max value {engine._calculated_max_value}")

    img = engine.get_current_bitmap_pil_image()
    print(f"Generated image size: {img.size}")
    img.save("test_engine_output.png")
    print("Saved test_engine_output.png")

    engine.randomize_parameters()
    engine.recalculate_pattern()
    img_rand = engine.get_current_bitmap_pil_image()
    img_rand.save("test_engine_random_output.png")
    print("Saved test_engine_random_output.png")

    # Test save/load (requires a dummy file to be created by file_io tests first, or adapt)
    # For standalone test, let's save then load
    test_save_path = "engine_test_file.chl"
    print(f"Saving to {test_save_path}...")
    engine.save_to_file(test_save_path, save_level_map=True)

    print(f"Loading from {test_save_path}...")
    new_engine = ChladniSimulator()
    new_engine.load_from_file(test_save_path)
    print(f"Loaded: {new_engine.width}x{new_engine.height}, {len(new_engine.wave_infos)} waves.")
    print(f"Wave 0 amplitude from loaded: {new_engine.wave_infos[0].amplitude if new_engine.wave_infos else 'N/A'}")
    assert new_engine.wave_infos[0].amplitude == engine.wave_infos[0].amplitude

    # Clean up
    import os
    if os.path.exists("test_engine_output.png"): os.remove("test_engine_output.png")
    if os.path.exists("test_engine_random_output.png"): os.remove("test_engine_random_output.png")
    if os.path.exists(test_save_path): os.remove(test_save_path)

    print("ChladniSimulator tests completed.")

# Need to import numpy for np.max in load_from_file if value_map is present
import numpy as np
