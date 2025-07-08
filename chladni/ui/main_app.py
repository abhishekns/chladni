import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import sys
import threading
import queue
import os

from PIL import Image, ImageTk # type: ignore

try:
    from ..chladni_engine import ChladniSimulator, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_CAPACITY, CHL_UNTITLED
    from ..visualization import DEFAULT_COLOR_MAPS
    from .dialogs import PropertiesDialog, AboutDialog
    from .settings_manager import SettingsManager
except ImportError:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from ..chladni_engine import ChladniSimulator, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_CAPACITY, CHL_UNTITLED
    from ..visualization import DEFAULT_COLOR_MAPS
    from .dialogs import PropertiesDialog, AboutDialog
    from .settings_manager import SettingsManager
    # Import constants for validation if needed, or handle in simulator
    from core import MIN_AMPLITUDE, MAX_AMPLITUDE, MIN_FREQ_RATIO, MAX_FREQ_RATIO, MIN_ANGLE, MAX_ANGLE


class ChladniApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.settings_manager = SettingsManager()

        self.last_open_dir = os.path.expanduser("~")
        self.last_save_dir = os.path.expanduser("~")
        self.last_export_dir = os.path.expanduser("~")

        self.load_initial_settings()

        self.simulator = ChladniSimulator()
        self.apply_loaded_settings_to_simulator()

        self._image_tk: ImageTk.PhotoImage | None = None
        self.render_thread: threading.Thread | None = None
        self.render_queue: queue.Queue = queue.Queue()
        self.stop_render_event: threading.Event = threading.Event()

        self.status_message_var = tk.StringVar(value="Ready")
        self.status_imgsize_var = tk.StringVar(value=f"{self.simulator.width}x{self.simulator.height}")
        self.status_zoom_var = tk.StringVar(value="100%")

        self._treeview_edit_entry: ttk.Entry | None = None # For in-place cell editing
        self._treeview_edit_item_id: str | None = None
        self._treeview_edit_column_id: str | None = None

        self.create_widgets()
        self.update_title()
        self.update_display_image()
        self.update_wave_grid()
        self.update_statusbar_imgsize()
        self.update_colormap_selection_from_simulator()
        self.normalize_var_menu.set(self.simulator.normalize)

        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        print("ChladniApp initialized.")

    def load_initial_settings(self):
        geom = self.settings_manager.get_setting('Window', 'geometry')
        if geom:
            try: self.root.geometry(geom)
            except tk.TclError: print(f"Warning: Invalid window geometry in settings: {geom}")
        else:
            initial_window_width = DEFAULT_WIDTH + 430
            initial_window_height = DEFAULT_HEIGHT + 150
            self.root.geometry(f"{initial_window_width}x{initial_window_height}")

        self.last_open_dir = self.settings_manager.get_setting('Window', 'last_open_dir', self.last_open_dir)
        self.last_save_dir = self.settings_manager.get_setting('Window', 'last_save_dir', self.last_save_dir)
        self.last_export_dir = self.settings_manager.get_setting('Window', 'last_export_dir', self.last_export_dir)

    def apply_loaded_settings_to_simulator(self):
        def_width = self.settings_manager.get_int_setting('Simulation', 'default_width', DEFAULT_WIDTH)
        def_height = self.settings_manager.get_int_setting('Simulation', 'default_height', DEFAULT_HEIGHT)
        self.simulator.set_dimensions(def_width, def_height)
        def_capacity = self.settings_manager.get_int_setting('Simulation', 'default_capacity', DEFAULT_CAPACITY)
        self.simulator.set_capacity(def_capacity)
        self.simulator.normalize = self.settings_manager.get_bool_setting('Simulation', 'default_normalize', True)
        def_colormap = self.settings_manager.get_setting('Simulation', 'default_colormap', "Spectrum")
        if def_colormap in self.simulator.available_color_maps:
            self.simulator.selected_color_map_name = def_colormap

    def update_title(self):
        base_title = "Chladni Plate Simulator (Python)"
        filename_part = os.path.basename(self.simulator.filename) if self.simulator.filename and self.simulator.filename != CHL_UNTITLED else "Untitled"
        modified_star = "*" if self.simulator.modified else ""
        self.root.title(f"{modified_star}{filename_part} - {base_title}")
        self.set_status_message(f"File: {filename_part}{modified_star}")

    def set_status_message(self, message: str): self.status_message_var.set(message)
    def update_statusbar_imgsize(self): self.status_imgsize_var.set(f"{self.simulator.width}x{self.simulator.height}")

    def create_widgets(self):
        main_frame = ttk.Frame(self.root, padding="5"); main_frame.pack(expand=True, fill=tk.BOTH)
        self.toolbar_frame = ttk.Frame(main_frame); self.toolbar_frame.pack(side=tk.TOP, fill=tk.X, pady=(0,5))
        self.render_button = ttk.Button(self.toolbar_frame, text="Render", command=self.on_cmd_render); self.render_button.pack(side=tk.LEFT, padx=2)
        self.randomize_button = ttk.Button(self.toolbar_frame, text="Randomize", command=self.on_cmd_randomize); self.randomize_button.pack(side=tk.LEFT, padx=2)
        self.stop_button = ttk.Button(self.toolbar_frame, text="Stop", command=self.on_cmd_stop_render, state=tk.DISABLED); self.stop_button.pack(side=tk.LEFT, padx=2)

        self.statusbar_frame = ttk.Frame(main_frame, relief=tk.SUNKEN, borderwidth=1); self.statusbar_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0))
        self.status_message_label = ttk.Label(self.statusbar_frame, textvariable=self.status_message_var, anchor=tk.W, relief=tk.FLAT); self.status_message_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        self.status_zoom_label = ttk.Label(self.statusbar_frame, textvariable=self.status_zoom_var, anchor=tk.E, relief=tk.FLAT); self.status_zoom_label.pack(side=tk.RIGHT, padx=5)
        self.status_imgsize_label = ttk.Label(self.statusbar_frame, textvariable=self.status_imgsize_var, anchor=tk.E, relief=tk.FLAT); self.status_imgsize_label.pack(side=tk.RIGHT, padx=5)

        content_frame = ttk.Frame(main_frame); content_frame.pack(expand=True, fill=tk.BOTH)
        self.left_panel = ttk.Frame(content_frame, width=285, relief=tk.SUNKEN, borderwidth=1); self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0,5))
        self.wave_grid_frame = ttk.Frame(self.left_panel); self.wave_grid_frame.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)
        self.wave_grid = ttk.Treeview(self.wave_grid_frame, columns=("on", "amplitude", "frequency", "phase"), show="headings")
        self.wave_grid.heading("on", text="On"); self.wave_grid.column("on", width=40, anchor=tk.CENTER)
        self.wave_grid.heading("amplitude", text="Amplitude"); self.wave_grid.column("amplitude", width=70, anchor=tk.E)
        self.wave_grid.heading("frequency", text="Frequency"); self.wave_grid.column("frequency", width=70, anchor=tk.E)
        self.wave_grid.heading("phase", text="Phase"); self.wave_grid.column("phase", width=70, anchor=tk.E)
        wave_grid_scrollbar = ttk.Scrollbar(self.wave_grid_frame, orient="vertical", command=self.wave_grid.yview); self.wave_grid.configure(yscrollcommand=wave_grid_scrollbar.set)
        wave_grid_scrollbar.pack(side=tk.RIGHT, fill=tk.Y); self.wave_grid.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.wave_grid.bind("<Double-1>", self._on_wave_grid_double_click)
        self.left_panel.pack_propagate(False)

        self.right_panel = ttk.Frame(content_frame, width=180, relief=tk.SUNKEN, borderwidth=1); self.right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5,0))
        ttk.Label(self.right_panel, text="Color Maps").pack(pady=5, anchor=tk.W, padx=5)
        self.colormap_listbox = tk.Listbox(self.right_panel, exportselection=False); self.colormap_listbox.pack(expand=True, fill=tk.BOTH, padx=5, pady=(0,5))
        for map_name in self.simulator.available_color_maps.keys(): self.colormap_listbox.insert(tk.END, map_name)
        self.colormap_listbox.bind('<<ListboxSelect>>', self.on_colormap_selected)
        self.right_panel.pack_propagate(False)

        self.center_panel = ttk.Frame(content_frame, relief=tk.SUNKEN, borderwidth=1); self.center_panel.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        self.image_label = ttk.Label(self.center_panel); self.image_label.pack(expand=True, padx=5, pady=5)

        self.menubar = tk.Menu(self.root); self.root.config(menu=self.menubar)
        file_menu = tk.Menu(self.menubar, tearoff=0); self.menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="New", command=self.on_file_new, accelerator="Ctrl+N")
        file_menu.add_command(label="Open...", command=self.on_file_open, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Save", command=self.on_file_save, accelerator="Ctrl+S")
        file_menu.add_command(label="Save As...", command=self.on_file_save_as, accelerator="Shift+Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Export Image...", command=self.on_file_export_image) # Connect to new method
        file_menu.add_separator(); file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_exit)

        self.root.bind_all("<Control-n>", lambda event: self.on_file_new())
        self.root.bind_all("<Control-o>", lambda event: self.on_file_open())
        self.root.bind_all("<Control-s>", lambda event: self.on_file_save())
        self.root.bind_all("<Control-S>", lambda event: self.on_file_save_as())

        commands_menu = tk.Menu(self.menubar, tearoff=0); self.menubar.add_cascade(label="Commands", menu=commands_menu)
        commands_menu.add_command(label="Render", command=self.on_cmd_render)
        commands_menu.add_command(label="Randomize", command=self.on_cmd_randomize)
        view_menu = tk.Menu(self.menubar, tearoff=0); self.menubar.add_cascade(label="View", menu=view_menu)
        self.normalize_var_menu = tk.BooleanVar(); view_menu.add_checkbutton(label="Normalize", variable=self.normalize_var_menu, command=self.on_view_normalize_toggle)
        view_menu.add_separator(); view_menu.add_command(label="Properties...", command=self.on_view_properties)
        help_menu = tk.Menu(self.menubar, tearoff=0); self.menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About Chladni Plate Simulator...", command=self.on_help_about)

    def update_colormap_selection_from_simulator(self):
        current_map_name = self.simulator.selected_color_map_name
        if current_map_name in self.simulator.available_color_maps:
            try:
                idx = list(self.simulator.available_color_maps.keys()).index(current_map_name)
                self.colormap_listbox.select_clear(0, tk.END); self.colormap_listbox.select_set(idx); self.colormap_listbox.see(idx)
            except ValueError: pass

    def update_display_image(self):
        pil_image = self.simulator.get_current_bitmap_pil_image()
        self._image_tk = ImageTk.PhotoImage(pil_image)
        self.image_label.config(image=self._image_tk)

    def update_wave_grid(self):
        for item in self.wave_grid.get_children(): self.wave_grid.delete(item)
        for i, wave_info in enumerate(self.simulator.wave_infos):
            on_display = "✓" if wave_info.on else "✗"
            self.wave_grid.insert("", tk.END, iid=str(i), values=(
                on_display, f"{wave_info.amplitude:.4f}", f"{wave_info.frequency:.4f}", f"{wave_info.phase:.1f}" ))

    def on_colormap_selected(self, event=None):
        selected_indices = self.colormap_listbox.curselection()
        if selected_indices:
            selected_map_name = self.colormap_listbox.get(selected_indices[0])
            if selected_map_name != self.simulator.selected_color_map_name:
                self.simulator.selected_color_map_name = selected_map_name
                self.update_display_image(); self.simulator.modified = True; self.update_title()

    def _ask_save_if_modified(self) -> bool:
        if self.simulator.modified:
            response = messagebox.askyesnocancel("Save Changes?", f"File '{os.path.basename(self.simulator.filename)}' has unsaved changes.\nDo you want to save them?")
            if response is True: return self.on_file_save()
            elif response is False: return True
            else: return False
        return True

    def on_file_new(self):
        if not self._ask_save_if_modified(): return
        self.simulator.reset(); self.apply_loaded_settings_to_simulator()
        self.update_title(); self.update_wave_grid(); self.update_display_image();
        self.update_statusbar_imgsize(); self.update_colormap_selection_from_simulator()
        self.normalize_var_menu.set(self.simulator.normalize)
        self.set_status_message("New file created.")

    def on_file_open(self):
        if not self._ask_save_if_modified(): return
        filepath = filedialog.askopenfilename(title="Open CHL File", initialdir=self.last_open_dir, filetypes=(("CHL files", "*.chl"), ("All files", "*.*")))
        if filepath:
            self.last_open_dir = os.path.dirname(filepath)
            if self.simulator.load_from_file(filepath):
                self.update_title(); self.update_wave_grid(); self.update_display_image();
                self.update_statusbar_imgsize(); self.update_colormap_selection_from_simulator()
                self.normalize_var_menu.set(self.simulator.normalize)
                self.set_status_message(f"File '{os.path.basename(filepath)}' loaded.")
            else:
                messagebox.showerror("Open Error", f"Failed to load file: {filepath}")
                self.set_status_message(f"Error loading file: {os.path.basename(filepath)}")

    def on_file_save(self) -> bool:
        if self.simulator.filename == CHL_UNTITLED or not os.path.exists(os.path.dirname(self.simulator.filename)):
            return self.on_file_save_as()
        else:
            if self.simulator.save_to_file(self.simulator.filename):
                self.update_title(); self.set_status_message(f"File saved to '{self.simulator.filename}'"); return True
            else:
                messagebox.showerror("Save Error", f"Failed to save file: {self.simulator.filename}")
                self.set_status_message(f"Error saving file: {self.simulator.filename}"); return False

    def on_file_save_as(self) -> bool:
        initial_dir = self.last_save_dir if os.path.isdir(self.last_save_dir) else os.path.dirname(self.simulator.filename) if self.simulator.filename and self.simulator.filename != CHL_UNTITLED else "."
        initial_file = os.path.basename(self.simulator.filename) if self.simulator.filename != CHL_UNTITLED else "untitled.chl"
        filepath = filedialog.asksaveasfilename(title="Save CHL File As", defaultextension=".chl", initialdir=initial_dir, initialfile=initial_file, filetypes=(("CHL files", "*.chl"), ("All files", "*.*")))
        if filepath:
            self.last_save_dir = os.path.dirname(filepath)
            if self.simulator.save_to_file(filepath):
                self.update_title(); self.set_status_message(f"File saved as '{filepath}'"); return True
            else:
                messagebox.showerror("Save As Error", f"Failed to save file: {filepath}")
                self.set_status_message(f"Error saving as file: {filepath}"); return False
        self.set_status_message("Save As cancelled."); return False

    def on_exit(self):
        if self._ask_save_if_modified():
            self.settings_manager.set_setting('Window', 'geometry', self.root.geometry())
            self.settings_manager.set_setting('Window', 'last_open_dir', self.last_open_dir)
            self.settings_manager.set_setting('Window', 'last_save_dir', self.last_save_dir)
            self.settings_manager.set_setting('Window', 'last_export_dir', self.last_export_dir)
            self.settings_manager.set_setting('Simulation', 'default_width', str(self.simulator.width))
            self.settings_manager.set_setting('Simulation', 'default_height', str(self.simulator.height))
            self.settings_manager.set_setting('Simulation', 'default_capacity', str(self.simulator.capacity))
            self.settings_manager.set_setting('Simulation', 'default_normalize', str(self.simulator.normalize))
            self.settings_manager.set_setting('Simulation', 'default_colormap', self.simulator.selected_color_map_name)
            self.settings_manager.save_settings()
            if self.render_thread and self.render_thread.is_alive(): self.stop_render_event.set()
            self.root.destroy()

    def on_file_export_image(self):
        if self.render_thread and self.render_thread.is_alive():
            messagebox.showwarning("Busy", "Cannot export image while rendering.")
            return
        pil_image = self.simulator.get_current_bitmap_pil_image()
        if not pil_image:
            messagebox.showerror("Export Error", "No image available to export.")
            return
        initial_filename = os.path.splitext(os.path.basename(self.simulator.filename if self.simulator.filename != CHL_UNTITLED else "chladni_pattern"))[0]
        filepath = filedialog.asksaveasfilename(
            title="Export Image As", initialdir=self.last_export_dir, initialfile=initial_filename, defaultextension=".png",
            filetypes=(("PNG files", "*.png"), ("JPEG files", "*.jpg;*.jpeg"), ("BMP files", "*.bmp"), ("All files", "*.*")))
        if filepath:
            self.last_export_dir = os.path.dirname(filepath)
            try:
                pil_image.save(filepath)
                self.set_status_message(f"Image exported to '{filepath}'")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export image: {e}")
                self.set_status_message(f"Error exporting image: {e}")
        else:
            self.set_status_message("Image export cancelled.")

    def _render_worker(self):
        try:
            self.simulator.recalculate_pattern_with_event(self.stop_render_event) # Pass the event
            if not self.stop_render_event.is_set(): self.render_queue.put("success")
            else: self.render_queue.put("stopped")
        except Exception as e: self.render_queue.put(e)

    def _check_render_queue(self):
        try:
            message = self.render_queue.get_nowait()
            if isinstance(message, Exception): self.on_render_complete(success=False, error=message)
            elif message == "success": self.update_display_image(); self.on_render_complete(success=True)
            elif message == "stopped": self.on_render_complete(success=False, stopped=True)
        except queue.Empty: self.root.after(100, self._check_render_queue)

    def on_render_start(self):
        self.root.config(cursor="watch")
        self.render_button.config(state=tk.DISABLED); self.randomize_button.config(state=tk.DISABLED); self.stop_button.config(state=tk.NORMAL)
        self.set_status_message("Rendering...")

    def on_render_complete(self, success: bool, error: Exception = None, stopped: bool = False):
        self.root.config(cursor="")
        self.render_button.config(state=tk.NORMAL); self.randomize_button.config(state=tk.NORMAL); self.stop_button.config(state=tk.DISABLED)
        self.render_thread = None; self.stop_render_event.clear()
        if success: msg = "Render complete."; self.simulator.modified = True; self.update_title()
        elif stopped: msg = "Render stopped by user."; self.set_status_message(msg)
        elif error: msg = f"Render failed: {error}"; messagebox.showerror("Render Error", str(error)); self.set_status_message(msg)
        else: msg = "Ready."; self.set_status_message(msg)
        print(msg)

    def on_cmd_render(self):
        if self.render_thread and self.render_thread.is_alive(): return
        self.on_render_start()
        self.render_thread = threading.Thread(target=self._render_worker, daemon=True); self.render_thread.start()
        self.root.after(100, self._check_render_queue)

    def on_cmd_stop_render(self):
        if self.render_thread and self.render_thread.is_alive():
            self.stop_render_event.set(); self.stop_button.config(state=tk.DISABLED); self.set_status_message("Stopping render...")
        else: print("No render process to stop.")

    def on_cmd_randomize(self):
        if self.render_thread and self.render_thread.is_alive():
            messagebox.showwarning("Busy", "Cannot randomize parameters while rendering."); return
        self.simulator.randomize_parameters(); self.update_wave_grid(); self.on_cmd_render()
        self.set_status_message("Parameters randomized. Rendering new pattern...")

    def on_view_normalize_toggle(self):
        new_normalize_state = self.normalize_var_menu.get()
        if self.simulator.normalize != new_normalize_state:
            self.simulator.normalize = new_normalize_state; self.simulator.modified = True
            self.update_title(); self.update_display_image()
            self.set_status_message(f"Normalization set to: {self.simulator.normalize}")

    def on_view_properties(self):
        if self.render_thread and self.render_thread.is_alive():
            messagebox.showwarning("Busy", "Cannot change properties while rendering."); return
        dialog = PropertiesDialog(self.root, self.simulator)
        if dialog.result:
            changed = False; props_changed_requires_rerender = False
            if self.simulator.width != dialog.result["width"] or self.simulator.height != dialog.result["height"]:
                self.simulator.set_dimensions(dialog.result["width"], dialog.result["height"])
                self.update_statusbar_imgsize(); changed = True; props_changed_requires_rerender = True
            if self.simulator.capacity != dialog.result["capacity"]:
                self.simulator.set_capacity(dialog.result["capacity"])
                self.update_wave_grid(); changed = True; props_changed_requires_rerender = True
            if self.simulator.normalize != dialog.result["normalize"]:
                self.simulator.normalize = dialog.result["normalize"]
                self.normalize_var_menu.set(self.simulator.normalize); changed = True
                self.update_display_image()
            if changed:
                self.simulator.modified = True; self.update_title()
                if props_changed_requires_rerender:
                    self.on_cmd_render(); self.set_status_message("Properties changed. Re-rendering pattern.")
                else: self.set_status_message("Properties updated.")
            else: self.set_status_message("Properties unchanged.")

    def on_help_about(self): AboutDialog(self.root)

    # --- Wave Grid Editing Methods ---
    def _destroy_cell_editor(self):
        if self._treeview_edit_entry:
            self._treeview_edit_entry.destroy()
            self._treeview_edit_entry = None
            self._treeview_edit_item_id = None
            self._treeview_edit_column_id = None

    def _cancel_cell_edit(self, event=None):
        self._destroy_cell_editor()

    def _apply_cell_edit(self, event=None):
        if not self._treeview_edit_entry or \
           self._treeview_edit_item_id is None or \
           self._treeview_edit_column_id is None:
            self._destroy_cell_editor()
            return

        new_value_str = self._treeview_edit_entry.get()
        wave_idx = int(self._treeview_edit_item_id)
        column_id = self._treeview_edit_column_id

        # print(f"Applying edit: wave_idx={wave_idx}, col={column_id}, val='{new_value_str}'")

        try:
            new_value_float = float(new_value_str)
            wave_info = self.simulator.wave_infos[wave_idx]

            original_params_for_undo_or_comparison = (wave_info.amplitude, wave_info.frequency, wave_info.phase)
            param_changed = False

            if column_id == "amplitude":
                clamped_value = max(MIN_AMPLITUDE, min(new_value_float, MAX_AMPLITUDE))
                if wave_info.amplitude != clamped_value:
                    wave_info.amplitude = clamped_value
                    param_changed = True
            elif column_id == "frequency":
                clamped_value = max(MIN_FREQ_RATIO, min(new_value_float, MAX_FREQ_RATIO))
                if wave_info.frequency != clamped_value:
                    wave_info.frequency = clamped_value
                    param_changed = True
            elif column_id == "phase":
                clamped_value = max(MIN_ANGLE, min(new_value_float, MAX_ANGLE))
                if wave_info.phase != clamped_value:
                    wave_info.phase = clamped_value
                    param_changed = True

            if param_changed:
                self.simulator.modified = True
                self.update_title()
                # Update only the specific item in treeview for efficiency
                on_display = "✓" if wave_info.on else "✗"
                self.wave_grid.item(str(wave_idx), values=(
                    on_display, f"{wave_info.amplitude:.4f}",
                    f"{wave_info.frequency:.4f}", f"{wave_info.phase:.1f}"
                ))
                # self.update_wave_grid() # Full refresh, less efficient

        except ValueError:
            messagebox.showerror("Invalid Input", f"Invalid value for {column_id}: '{new_value_str}'. Please enter a number.")
        except IndexError:
            messagebox.showerror("Error", "Invalid wave index.")
        finally:
            self._destroy_cell_editor()


    def _on_wave_grid_double_click(self, event):
        self._destroy_cell_editor() # Clear any previous editor

        region = self.wave_grid.identify_region(event.x, event.y)
        if region != "cell":
            return

        column_id_num_str = self.wave_grid.identify_column(event.x) # e.g., "#1", "#2"
        column_idx = int(column_id_num_str.replace("#", "")) -1 # Convert to 0-based index

        # Get column identifier string (e.g., "amplitude")
        # The `columns` tuple in Treeview definition is 0-indexed.
        # `self.wave_grid['columns']` gives ('on', 'amplitude', 'frequency', 'phase')
        column_id_str = self.wave_grid['columns'][column_idx]

        item_id = self.wave_grid.identify_row(event.y) # This is our wave index as string

        if not item_id: # Clicked on header or empty space
            return

        # print(f"Double click: item={item_id}, col_id_num={column_id_num_str}, col_idx={column_idx}, col_id_str='{column_id_str}'")

        # --- Handle "On/Off" column ---
        if column_id_str == "on":
            wave_idx = int(item_id)
            try:
                wave_info = self.simulator.wave_infos[wave_idx]
                wave_info.on = not wave_info.on
                self.simulator.modified = True
                self.update_title()
                # Update just this row in the grid
                on_display = "✓" if wave_info.on else "✗"
                self.wave_grid.item(item_id, values=(
                    on_display, f"{wave_info.amplitude:.4f}",
                    f"{wave_info.frequency:.4f}", f"{wave_info.phase:.1f}"
                ))
            except IndexError:
                messagebox.showerror("Error", "Invalid wave index for toggling 'on' state.")
            return # Done with "on" column

        # --- Handle Numerical Columns (Amplitude, Frequency, Phase) ---
        if column_id_str in ["amplitude", "frequency", "phase"]:
            x, y, width, height = self.wave_grid.bbox(item_id, column=column_id_num_str)

            # Get the raw, unformatted value from the simulator
            wave_idx = int(item_id)
            wave_info = self.simulator.wave_infos[wave_idx]
            current_value = ""
            if column_id_str == "amplitude": current_value = wave_info.amplitude
            elif column_id_str == "frequency": current_value = wave_info.frequency
            elif column_id_str == "phase": current_value = wave_info.phase

            entry_var = tk.StringVar(value=str(current_value))
            self._treeview_edit_entry = ttk.Entry(self.wave_grid_frame, textvariable=entry_var, justify=tk.RIGHT) # Use wave_grid_frame as parent
            self._treeview_edit_entry.place(x=x, y=y, width=width, height=height, anchor='nw')

            self._treeview_edit_item_id = item_id
            self._treeview_edit_column_id = column_id_str # Store the string ID

            self._treeview_edit_entry.focus_set()
            self._treeview_edit_entry.select_range(0, tk.END)

            self._treeview_edit_entry.bind("<Return>", self._apply_cell_edit)
            self._treeview_edit_entry.bind("<FocusOut>", self._apply_cell_edit) # Apply on lose focus
            self._treeview_edit_entry.bind("<Escape>", self._cancel_cell_edit)

    def run(self): self.root.mainloop()

def main():
    root = tk.Tk()
    app = ChladniApp(root)
    root.mainloop()

if __name__ == '__main__':
    main()
