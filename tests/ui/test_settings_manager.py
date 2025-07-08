import unittest
from unittest.mock import patch, mock_open, MagicMock
import configparser
import os

# Adjust import path for tests
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from chladni.ui.settings_manager import SettingsManager, APP_NAME

class TestSettingsManager(unittest.TestCase):

    @patch('os.path.exists')
    @patch('os.makedirs')
    @patch('configparser.ConfigParser.read')
    @patch('builtins.open', new_callable=mock_open)
    def _create_manager_for_test(self, mock_open_instance, mock_read, mock_makedirs, mock_exists,
                                 exists_returns=False, initial_config_data=None):
        """Helper to create a SettingsManager instance with mocked dependencies."""
        mock_exists.return_value = exists_returns

        # If initial_config_data is provided, mock_read should populate the config
        if initial_config_data:
            def side_effect_read(filepath):
                # This is a bit simplified; normally ConfigParser.read directly modifies 'self'
                # For testing, we'll have the mock_config object reflect these values.
                pass
            mock_read.side_effect = side_effect_read

        # The SettingsManager constructor calls self._get_config_filepath,
        # which might call os.makedirs if os.path.exists is false for the dir.
        # Then it calls self.load_settings().
        # If exists_returns is False for the file, _set_default_settings and save_settings are called.

        # We need to control the ConfigParser instance used by SettingsManager
        mock_config_parser_instance = configparser.ConfigParser()
        if initial_config_data:
            mock_config_parser_instance.read_dict(initial_config_data)

        with patch('configparser.ConfigParser', return_value=mock_config_parser_instance) as mock_cp_class:
            manager = SettingsManager(filename="test_settings.ini")
            # Store the mocked config instance on the manager for direct assertion if needed
            manager.config = mock_config_parser_instance
            return manager, mock_config_parser_instance


    @patch('os.name', 'nt')
    @patch('os.getenv')
    @patch('os.path.expanduser')
    @patch('os.path.join', side_effect=lambda *args: '/'.join(args)) # Simple join mock
    def test_get_config_filepath_windows(self, mock_expanduser, mock_getenv, mock_os_name):
        mock_expanduser.return_value = "C:/Users/testuser"
        mock_getenv.return_value = "C:/Users/testuser/AppData/Roaming"

        manager = SettingsManager(filename="test.ini") # Constructor calls _get_config_filepath
        expected_path = f"C:/Users/testuser/AppData/Roaming/{APP_NAME}/test.ini"
        self.assertEqual(manager.filepath, expected_path)

        # Test APPDATA not set
        mock_getenv.return_value = None
        manager_no_appdata = SettingsManager(filename="test.ini")
        expected_path_no_appdata = f"C:/Users/testuser/{APP_NAME}/test.ini"
        self.assertEqual(manager_no_appdata.filepath, expected_path_no_appdata)


    @patch('os.name', 'posix') # For Linux/macOS
    @patch('os.path.expanduser')
    @patch('os.path.join', side_effect=lambda *args: '/'.join(args))
    def test_get_config_filepath_linux_macos(self, mock_join, mock_expanduser, mock_os_name):
        mock_expanduser.return_value = "/home/testuser"
        manager = SettingsManager(filename="test.ini")
        expected_path = f"/home/testuser/.config/{APP_NAME}/test.ini"
        self.assertEqual(manager.filepath, expected_path)

    @patch('os.path.exists', return_value=False) # File does not exist
    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('configparser.ConfigParser.write') # Mock write for default settings save
    def test_load_settings_file_not_exists_creates_defaults(self, mock_write, mock_open_instance, mock_makedirs, mock_path_exists):
        manager, mock_config = self._create_manager_for_test(exists_returns=False)

        # Check that _set_default_settings was effectively called (by checking config values)
        self.assertTrue(mock_config.has_section('Window'))
        self.assertTrue(mock_config.has_section('Simulation'))
        self.assertEqual(mock_config.get('Simulation', 'default_width'), '500')
        # Check that save_settings was called
        mock_write.assert_called_once()


    def test_get_setting(self):
        initial_data = {'TestSection': {'testkey': 'testvalue'}}
        manager, mock_config = self._create_manager_for_test(exists_returns=True, initial_config_data=initial_data)

        self.assertEqual(manager.get_setting('TestSection', 'testkey'), 'testvalue')
        self.assertIsNone(manager.get_setting('TestSection', 'nonexistentkey'))
        self.assertEqual(manager.get_setting('TestSection', 'nonexistentkey', fallback='default'), 'default')
        self.assertIsNone(manager.get_setting('NonExistentSection', 'key'))


    def test_get_int_setting(self):
        initial_data = {'TestSection': {'intkey': '123', 'stringkey': 'abc', 'floatkey': '1.23'}}
        manager, mock_config = self._create_manager_for_test(exists_returns=True, initial_config_data=initial_data)

        self.assertEqual(manager.get_int_setting('TestSection', 'intkey', 0), 123)
        self.assertEqual(manager.get_int_setting('TestSection', 'stringkey', 0), 0) # Fallback on ValueError
        self.assertEqual(manager.get_int_setting('TestSection', 'floatkey', 0), 0) # Fallback on ValueError from getint
        self.assertEqual(manager.get_int_setting('TestSection', 'nonexistentkey', 42), 42)
        self.assertEqual(manager.get_int_setting('NonExistentSection', 'key', 99), 99)


    def test_get_bool_setting(self):
        initial_data = {'TestSection': {'true_val1': 'true', 'true_val2': 'YES', 'true_val3': '1',
                                      'false_val1': 'false', 'false_val2': 'NO', 'false_val3': '0',
                                      'stringkey': 'abc'}}
        manager, mock_config = self._create_manager_for_test(exists_returns=True, initial_config_data=initial_data)

        self.assertTrue(manager.get_bool_setting('TestSection', 'true_val1', False))
        self.assertTrue(manager.get_bool_setting('TestSection', 'true_val2', False))
        self.assertTrue(manager.get_bool_setting('TestSection', 'true_val3', False))
        self.assertFalse(manager.get_bool_setting('TestSection', 'false_val1', True))
        self.assertFalse(manager.get_bool_setting('TestSection', 'false_val2', True))
        self.assertFalse(manager.get_bool_setting('TestSection', 'false_val3', True))

        self.assertTrue(manager.get_bool_setting('TestSection', 'stringkey', True)) # Fallback on ValueError
        self.assertFalse(manager.get_bool_setting('TestSection', 'nonexistentkey', False))
        self.assertTrue(manager.get_bool_setting('NonExistentSection', 'key', True))

    def test_set_setting(self):
        manager, mock_config = self._create_manager_for_test(exists_returns=False) # Start with empty config

        manager.set_setting('NewSection', 'newkey', 'newvalue')
        self.assertTrue(mock_config.has_section('NewSection'))
        self.assertEqual(mock_config.get('NewSection', 'newkey'), 'newvalue')

        manager.set_setting('NewSection', 'newkey', 'updatedvalue')
        self.assertEqual(mock_config.get('NewSection', 'newkey'), 'updatedvalue')

        manager.set_setting('NewSection', 'anotherkey', 123) # Should convert value to string
        self.assertEqual(mock_config.get('NewSection', 'anotherkey'), '123')

    @patch('builtins.open', new_callable=mock_open)
    def test_save_settings(self, mock_open_instance):
        manager, mock_config = self._create_manager_for_test(exists_returns=False) # Initializes with default config
        mock_config.set('TestSection', 'key', 'value')

        manager.save_settings()

        mock_open_instance.assert_called_once_with(manager.filepath, 'w')
        # config.write should have been called on the file handle from open
        mock_config.write.assert_called_once_with(mock_open_instance())

    @patch('builtins.open', side_effect=IOError("Permission denied"))
    @patch('sys.stdout', new_callable=MagicMock) # To capture print
    def test_save_settings_io_error(self, mock_stdout, mock_open_error):
        # Use a real config parser instance here as we are not mocking its write method directly,
        # but testing the exception handling in SettingsManager.save_settings
        with patch('configparser.ConfigParser.write') as mock_config_write:
            # This mock_config_write is for the default settings save during init.
            # The actual save_settings call will use the real config.write.
            manager = SettingsManager(filename="test_settings.ini")

        manager.set_setting("S", "K", "V") # Make some change to ensure config is written
        manager.save_settings() # This should trigger the IOError

        # Check that a warning was printed (or logged)
        # This is a bit fragile as it depends on the exact print message.
        # A more robust way would be to use logging and assert logging calls.
        self.assertTrue(any("Warning: Could not save settings" in call_args[0][0] for call_args in mock_stdout.write.call_args_list))


if __name__ == '__main__':
    unittest.main()
