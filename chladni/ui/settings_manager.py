import configparser
import os

APP_NAME = "ChladniPy" # Used for settings directory

class SettingsManager:
    def __init__(self, filename="settings.ini"):
        self.config = configparser.ConfigParser()
        self.filepath = self._get_config_filepath(filename)
        self.load_settings()

    def _get_config_filepath(self, filename: str) -> str:
        # Platform-specific directory for application settings
        if os.name == 'nt': # Windows
            app_data_dir = os.getenv('APPDATA')
            if not app_data_dir: # Fallback if APPDATA is not set
                 app_data_dir = os.path.expanduser("~")
        else: # Linux, macOS, etc.
            app_data_dir = os.path.join(os.path.expanduser("~"), ".config")

        settings_dir = os.path.join(app_data_dir, APP_NAME)
        if not os.path.exists(settings_dir):
            try:
                os.makedirs(settings_dir)
            except OSError as e:
                print(f"Warning: Could not create settings directory {settings_dir}: {e}")
                # Fallback to current directory if settings dir creation fails
                return os.path.join(".", filename)
        return os.path.join(settings_dir, filename)

    def load_settings(self):
        if os.path.exists(self.filepath):
            self.config.read(self.filepath)
        else:
            # Create default settings if file doesn't exist
            self._set_default_settings()
            self.save_settings() # Save them so file is created

    def _set_default_settings(self):
        self.config['Window'] = {
            'geometry': '', # Let system decide initially or provide a sensible default like '800x600+100+100'
            'last_open_dir': os.path.expanduser("~"),
            'last_save_dir': os.path.expanduser("~"),
            'last_export_dir': os.path.expanduser("~")
        }
        self.config['Simulation'] = {
            'default_width': '500',
            'default_height': '500',
            'default_capacity': '10',
            'default_normalize': 'True',
            'default_colormap': 'Spectrum' # Name of the colormap
        }

    def save_settings(self):
        try:
            with open(self.filepath, 'w') as configfile:
                self.config.write(configfile)
        except IOError as e:
            print(f"Warning: Could not save settings to {self.filepath}: {e}")


    def get_setting(self, section, key, fallback=None):
        return self.config.get(section, key, fallback=fallback)

    def get_int_setting(self, section, key, fallback=0):
        try:
            return self.config.getint(section, key)
        except (configparser.NoOptionError, configparser.NoSectionError, ValueError):
            return fallback

    def get_bool_setting(self, section, key, fallback=False):
        try:
            return self.config.getboolean(section, key)
        except (configparser.NoOptionError, configparser.NoSectionError, ValueError):
            return fallback

    def set_setting(self, section, key, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, str(value))

if __name__ == '__main__':
    # Test settings manager
    sm = SettingsManager("test_settings.ini")
    print(f"Settings file path: {sm.filepath}")

    # Set some test values
    sm.set_setting("Window", "geometry", "900x700+50+50")
    sm.set_setting("Simulation", "default_width", "600")
    sm.save_settings()

    # Reload and check
    sm2 = SettingsManager("test_settings.ini")
    print(f"Loaded geometry: {sm2.get_setting('Window', 'geometry')}")
    print(f"Loaded default_width: {sm2.get_int_setting('Simulation', 'default_width', 500)}")

    # Clean up test file
    if os.path.exists(sm.filepath):
        os.remove(sm.filepath)
        # Try to remove directory if it was the test one
        if os.path.basename(os.path.dirname(sm.filepath)) == APP_NAME and len(os.listdir(os.path.dirname(sm.filepath))) == 0 :
             try: os.rmdir(os.path.dirname(sm.filepath))
             except OSError: pass # Might fail if it's a real settings dir used by another test
        print("Test settings file cleaned up.")
