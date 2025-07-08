import argparse
import os
from .core import calculate_chladni_pattern, ValueMap, WaveInfo, ITERATION_MULTIPLIER
from .file_io import load_chl_file, save_chl_file, ChladniData
from .visualization import DEFAULT_COLOR_MAPS, generate_bitmap_pil

def main():
    parser = argparse.ArgumentParser(description="Chladni Plate Simulator CLI")
    parser.add_argument("input_chl_file", help="Path to the input .chl file")
    parser.add_argument("-o", "--output_image", help="Path to save the output PNG image (e.g., output.png)")
    parser.add_argument("-s", "--output_chl", help="Path to save the output .chl file (e.g., output.chl). If specified, the calculated ValueMap will be included.")
    parser.add_argument("-c", "--colormap", help="Name of the colormap to use for the image.", choices=DEFAULT_COLOR_MAPS.keys(), default="Spectrum")
    # parser.add_argument("--width", type=int, help="Override width from .chl file.") # Future enhancement
    # parser.add_argument("--height", type=int, help="Override height from .chl file.") # Future enhancement

    args = parser.parse_args()

    if not os.path.exists(args.input_chl_file):
        print(f"Error: Input file '{args.input_chl_file}' not found.")
        return

    print(f"Loading CHL file: {args.input_chl_file}")
    try:
        chl_data = load_chl_file(args.input_chl_file)
    except Exception as e:
        print(f"Error loading CHL file: {e}")
        return

    print(f"Loaded: {len(chl_data.wave_infos)} wave infos, Image Size: {chl_data.width}x{chl_data.height}, Normalize: {chl_data.normalize}")

    # Prepare ValueMap for calculation
    # If .chl file contained a value_map, we could use its dimensions, but calculation will overwrite it.
    # The Chladni class in Pascal seems to always calculate if you call CalcLevelMap.
    # It doesn't just use the loaded map unless only MakeBitmap is called.
    # For the CLI, we will always recalculate.

    simulation_value_map = ValueMap(chl_data.width, chl_data.height)

    print("Calculating Chladni pattern...")
    # The calculate_chladni_pattern function returns the max_val which can be used for normalization
    # The chl_data.normalize flag from the file indicates if the *original* rendering was normalized
    # when it was saved, or if the color map should be normalized to the data when displayed.
    # TCladni.MakeBitmap:
    #   if (fNormalize and (fValMax > 0)) then map.MaxIter := fValMax else map.MaxIter := ITERATION_MULTIPLIER;
    # This means fValMax is the result of the last CalcLevelMap.

    calculated_max_value = calculate_chladni_pattern(
        simulation_value_map,
        chl_data.wave_infos,
        chl_data.width,
        chl_data.height
    )
    print(f"Calculation complete. Max value from calculation: {calculated_max_value}")

    if args.output_image:
        if args.colormap not in DEFAULT_COLOR_MAPS:
            print(f"Error: Colormap '{args.colormap}' not found. Available: {', '.join(DEFAULT_COLOR_MAPS.keys())}")
            return

        selected_color_map = DEFAULT_COLOR_MAPS[args.colormap]

        print(f"Generating image with colormap '{selected_color_map.name}'...")
        # The `chl_data.normalize` flag from the input file determines if normalization should be applied
        # using the `calculated_max_value`.
        pil_image = generate_bitmap_pil(
            simulation_value_map,
            selected_color_map,
            current_max_value_in_map=calculated_max_value,
            normalize=chl_data.normalize # Use normalize flag from input .chl
        )

        try:
            pil_image.save(args.output_image)
            print(f"Output image saved to: {args.output_image}")
        except Exception as e:
            print(f"Error saving image: {e}")

    if args.output_chl:
        print(f"Saving output CHL file to: {args.output_chl}")
        # Create a new ChladniData object for saving.
        # It should contain the original wave_infos, map_index, width, height, normalize flag,
        # but with the newly calculated value_map.

        output_chl_data = ChladniData(
            wave_infos=chl_data.wave_infos,
            map_index=chl_data.map_index, # Preserve original map_index
            width=chl_data.width,
            height=chl_data.height,
            normalize=chl_data.normalize, # Preserve original normalize flag
            value_map=simulation_value_map, # The newly calculated map
            filename=args.output_chl
        )

        try:
            # Save with the level map included
            save_chl_file(args.output_chl, output_chl_data, save_level_map=True)
            print(f"Output CHL file saved to: {args.output_chl}")
        except Exception as e:
            print(f"Error saving CHL file: {e}")

    print("CLI processing finished.")

if __name__ == "__main__":
    main()
