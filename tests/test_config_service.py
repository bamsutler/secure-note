import unittest
from unittest.mock import patch, MagicMock, mock_open
import yaml
from pathlib import Path
import os

from src.config_service import ConfigurationService

class TestConfigurationService(unittest.TestCase):

    def _get_default_test_config(self):
        # A simplified version of the actual default config for easier testing
        return {
            'database': {'name': '/test/path/transcriptions.db'},
            'paths': {'markdown_save': '/test/path/analyses'},
            'models': {'whisper': {'default': 'tiny'}}
        }

    @patch('src.config_service.Path.home')
    @patch('src.config_service.Path.mkdir')
    @patch('src.config_service.yaml.dump')
    @patch('src.config_service.yaml.safe_load')
    @patch('src.config_service.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_from_local_config_file(self, mock_open_m, mock_path_exists, mock_yaml_safe_load, mock_yaml_dump, mock_mkdir, mock_path_home):
        """Test loading configuration from a local config.yaml file."""
        mock_path_home.return_value = Path("/fake/home")
        local_config_path = Path("config.yaml")
        
        # Path.exists calls:
        # 1. In _determine_config_path on local_config_path (should be True)
        # 2. In _load_config on self._config_path (which is local_config_path, should be True)
        mock_path_exists.side_effect = [True, True]
        
        test_config_data = self._get_default_test_config()
        mock_yaml_safe_load.return_value = test_config_data

        service = ConfigurationService(config_file_name="config.yaml", app_name="test_app")
        loaded_config = service.get_config()

        self.assertEqual(mock_path_exists.call_count, 2) 
        mock_open_m.assert_called_once_with(local_config_path, 'r')
        mock_yaml_safe_load.assert_called_once()
        self.assertEqual(loaded_config, test_config_data)
        mock_yaml_dump.assert_not_called() # Should not save defaults if loaded

    @patch('src.config_service.Path.home')
    @patch('src.config_service.Path.mkdir')
    @patch('src.config_service.yaml.dump')
    @patch('src.config_service.yaml.safe_load')
    @patch('src.config_service.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    def test_load_from_home_config_file(self, mock_open_m, mock_path_exists, mock_yaml_safe_load, mock_yaml_dump, mock_mkdir, mock_path_home):
        """Test loading configuration from a config file in the home directory."""
        fake_home_dir = Path("/fake/home")
        app_name = "test_app_home"
        config_file = "custom_config.yaml"
        mock_path_home.return_value = fake_home_dir
        home_app_config_dir = fake_home_dir / f".{app_name}"
        home_config_file_path = home_app_config_dir / config_file

        # Simulate local config.yaml does NOT exist (1st call to exists()), 
        # but home one does (2nd call to exists())
        mock_path_exists.side_effect = [False, True]

        test_config_data = {'paths': {'temp': '/home/temp'}}
        mock_yaml_safe_load.return_value = test_config_data

        service = ConfigurationService(config_file_name=config_file, app_name=app_name)
        loaded_config = service.get_config()

        self.assertEqual(mock_path_exists.call_count, 2)
        # First call to exists() is on Path(config_file), second on home_config_file_path
        # We can inspect mock_path_exists.mock_calls if needed, but call_count and side_effect cover the sequence.

        mock_mkdir.assert_any_call(parents=True, exist_ok=True) # For home_app_config_dir.mkdir call
        mock_open_m.assert_called_once_with(home_config_file_path, 'r')
        mock_yaml_safe_load.assert_called_once()
        self.assertEqual(loaded_config, test_config_data)
        mock_yaml_dump.assert_not_called()

    @patch('src.config_service.Path.home')
    @patch('src.config_service.Path.mkdir')
    @patch('src.config_service.yaml.dump')
    @patch('src.config_service.yaml.safe_load')
    @patch('src.config_service.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.config_service.ConfigurationService._get_default_config') # Mock the actual default generator
    def test_load_defaults_when_no_config_file_exists(self, mock_get_defaults, mock_open_m, mock_path_exists, mock_yaml_safe_load, mock_yaml_dump, mock_mkdir, mock_path_home):
        """Test loading default configuration when no config file is found and saves it."""
        fake_home_dir = Path("/fake/home")
        app_name = "new_app"
        config_file = "new_config.yaml"
        mock_path_home.return_value = fake_home_dir
        home_app_config_dir = fake_home_dir / f".{app_name}"
        expected_home_config_path = home_app_config_dir / config_file # Path where it checks for home config and saves defaults

        mock_path_exists.return_value = False # Simulate no config files exist anywhere

        default_config_data = self._get_default_test_config()
        mock_get_defaults.return_value = default_config_data

        service = ConfigurationService(config_file_name=config_file, app_name=app_name)
        loaded_config = service.get_config()

        # Check that Path.exists was called for local and home paths
        self.assertEqual(mock_path_exists.call_count, 2, "Path.exists should be called twice (local and home)")

        mock_get_defaults.assert_called_once()
        self.assertEqual(loaded_config, default_config_data)
        
        # Check that it tries to save the defaults to the home config path
        mock_mkdir.assert_any_call(parents=True, exist_ok=True) 
        mock_open_m.assert_called_once_with(expected_home_config_path, 'w')
        mock_yaml_dump.assert_called_once_with(default_config_data, mock_open_m(), default_flow_style=False)
        mock_yaml_safe_load.assert_not_called() # Should not load if no file initially

    @patch('src.config_service.Path.home')
    @patch('src.config_service.ConfigurationService._get_default_config')
    def test_get_method_retrieves_values(self, mock_get_defaults, mock_path_home):
        """Test the get() method for retrieving specific config values."""
        mock_path_home.return_value = Path("/fake/home")
        # Simulate config is loaded with some data (e.g., defaults)
        test_config = {
            'database': {'name': 'mydb.db', 'host': 'localhost'},
            'feature': {'enabled': True, 'level': 5}
        }
        mock_get_defaults.return_value = test_config

        # Patch Path.exists, open, and Path.mkdir to avoid file/dir operations
        with patch('src.config_service.Path.exists', return_value=False), \
             patch('builtins.open', mock_open()), \
             patch('src.config_service.Path.mkdir') as mock_path_mkdir: # Added patch for Path.mkdir
            service = ConfigurationService() # Will load mocked defaults

        # Now service.config should be test_config due to mock_get_defaults
        self.assertEqual(service.get('database', 'name'), 'mydb.db')
        self.assertEqual(service.get('feature', 'level'), 5)
        self.assertTrue(service.get('feature', 'enabled'))
        self.assertIsNone(service.get('nonexistent', 'key'))
        self.assertEqual(service.get('feature', 'missing', default='default_val'), 'default_val')
        
        # Verify that the directory creation for home_dir_config_path was attempted
        # because Path.exists returned False for all config files.
        # The mkdir call would be on home_dir_config_path.parent
        self.assertTrue(mock_path_mkdir.called)

    @patch('src.config_service.Path.home')
    @patch('src.config_service.Path.mkdir')
    @patch('src.config_service.yaml.dump')
    @patch('src.config_service.yaml.safe_load')
    @patch('src.config_service.Path.exists')
    @patch('builtins.open', new_callable=mock_open)
    @patch('src.config_service.ConfigurationService._get_default_config')
    def test_empty_or_invalid_yaml_loads_defaults(self, mock_get_defaults, mock_open_m, mock_path_exists, mock_yaml_safe_load, mock_yaml_dump, mock_mkdir, mock_path_home):
        """Test that if a config file exists but is empty/invalid, defaults are loaded and saved."""
        mock_path_home.return_value = Path("/fake/home")
        local_config_path = Path("config.yaml")
        
        mock_path_exists.return_value = True # Simulate local config.yaml exists
        mock_yaml_safe_load.return_value = None # Simulate empty or invalid YAML content
        
        default_config_data = self._get_default_test_config()
        mock_get_defaults.return_value = default_config_data

        service = ConfigurationService(config_file_name="config.yaml", app_name="test_app_invalid")
        loaded_config = service.get_config()

        mock_open_m.assert_any_call(local_config_path, 'r') # Attempt to read
        mock_yaml_safe_load.assert_called_once()
        mock_get_defaults.assert_called_once()
        self.assertEqual(loaded_config, default_config_data)
        
        # Check that it tries to save the defaults over the invalid local one
        # The _config_path will be the local one in this case
        mock_open_m.assert_any_call(local_config_path, 'w') 
        mock_yaml_dump.assert_called_once_with(default_config_data, mock_open_m(), default_flow_style=False)

if __name__ == '__main__':
    unittest.main() 