import unittest
from unittest.mock import patch, MagicMock, PropertyMock, call
import numpy as np
import threading # For mocking threading.Thread if AudioService uses it directly for callbacks, which it doesn't seem to for PyAudio streams.

from src.audio_service import AudioService
from src.config_service import ConfigurationService
import pyaudio # Import for type hinting and potentially for PyAudio specific exceptions if needed

class TestAudioService(unittest.TestCase):

    def _get_config_side_effect(self, *args, **kwargs):
        if args == ('audio', 'frames_per_buffer'): # Changed from chunk_size to frames_per_buffer
            return 1024
        if args == ('audio', 'samplerate'): # Changed from rate to samplerate
            return 16000
        if args == ('audio', 'channels'):
            return 1
        if args == ('audio', 'format'):
            return 'paFloat32'
        return MagicMock()

    @patch('src.audio_service.pyaudio.PyAudio') # Patch PyAudio class where it's instantiated in AudioService
    def setUp(self, mock_pyaudio_class):
        self.mock_config_service = MagicMock(spec=ConfigurationService)
        self.mock_config_service.get.side_effect = self._get_config_side_effect
        
        self.mock_pa_instance = MagicMock()
        mock_pyaudio_class.return_value = self.mock_pa_instance
        self.mock_pa_instance.get_device_count.return_value = 0
        self.mock_pa_instance.get_default_input_device_info.return_value = {
            'index': 0, 'name': 'Default PA Input', 'maxInputChannels': 1, 'defaultSampleRate': 44100.0, 'hostApi': 0
        }
        self.mock_pa_instance.get_default_output_device_info.return_value = {
            'index': 1, 'name': 'Default PA Output', 'maxOutputChannels': 2, 'defaultSampleRate': 44100.0, 'hostApi': 0
        }
        # Mock get_host_api_info_by_index for list_input_devices
        self.mock_pa_instance.get_host_api_info_by_index.return_value = {'name': 'Core Audio'}

        # Instantiate AudioService with the mocked ConfigurationService
        # AudioService itself will now use the mocked PyAudio instance via mock_pyaudio_class
        self.audio_service = AudioService(config_service=self.mock_config_service) # Assuming AudioService takes config
        # Re-assign pa_instance to the one AudioService would have created if we want to assert on it directly
        # However, AudioService creates its own, so we assert on the mock_pa_instance that it *would* have used.
        # The mock_pyaudio_class.return_value ensures our self.mock_pa_instance IS the one used by AudioService.

    def tearDown(self):
        # AudioService now has its own close method that should terminate its PyAudio instance
        self.audio_service.close() 
        # We can assert that terminate was called on the instance AudioService holds
        # which is self.mock_pa_instance due to the setUp patching.
        if self.audio_service.pyaudio_instance is not None: # If not already set to None by close()
             self.mock_pa_instance.terminate.assert_called_once()
        elif self.mock_pa_instance.terminate.called: # if close() set it to None, check if it was called before that
            pass # Already asserted by implication of being None after close
        # else: # This case implies terminate was not called and instance is None for other reasons (problematic)
            # self.fail("PyAudio instance was None but terminate was not called, or close() failed to call terminate")

    def test_initialization(self):
        self.assertIsNotNone(self.audio_service)
        self.assertEqual(self.audio_service.frames_per_buffer, 1024)
        self.assertEqual(self.audio_service.channels, 1)
        self.assertEqual(self.audio_service.py_format, pyaudio.paFloat32)
        self.mock_pa_instance.terminate.assert_not_called() # Should not terminate on init

    def test_list_input_devices_no_devices(self):
        self.mock_pa_instance.get_device_count.return_value = 0
        devices = self.audio_service.list_input_devices()
        self.assertEqual(devices, [])

    def test_list_input_devices_with_pyaudio_devices(self):
        pa_device_list_info = [
            {'index': 0, 'name': 'PA Mic 1', 'maxInputChannels': 1, 'defaultSampleRate': 48000.0, 'hostApi': 0},
            {'index': 1, 'name': 'PA Line In', 'maxInputChannels': 2, 'defaultSampleRate': 44100.0, 'hostApi': 0},
            {'index': 2, 'name': 'PA Output Device', 'maxInputChannels': 0, 'defaultSampleRate': 44100.0, 'hostApi': 0} # Not an input
        ]
        self.mock_pa_instance.get_device_count.return_value = len(pa_device_list_info)
        self.mock_pa_instance.get_device_info_by_index.side_effect = lambda i: pa_device_list_info[i]
        self.mock_pa_instance.get_host_api_info_by_index.return_value = {'name': 'TestAPI'}

        devices = self.audio_service.list_input_devices()
        
        self.assertEqual(len(devices), 2) # Only input devices
        self.assertEqual(devices[0]['name'], 'PA Mic 1')
        self.assertEqual(devices[1]['name'], 'PA Line In')
        self.assertEqual(devices[0]['host_api_name'], 'TestAPI')

    def test_start_recording_mic_only_success(self):
        mock_mic_stream = MagicMock()
        self.mock_pa_instance.open.return_value = mock_mic_stream
        # self.mock_pa_instance.get_default_input_device_info.return_value = {'index': 0, ... } # Already in setUp

        result = self.audio_service.start_recording(samplerate=16000, mic_device_index=0, include_system_audio=False)
        
        self.assertTrue(result)
        self.mock_pa_instance.open.assert_called_once_with(
            format=self.audio_service.py_format, channels=self.audio_service.channels, rate=16000,
            input=True, input_device_index=0,
            frames_per_buffer=self.audio_service.frames_per_buffer, stream_callback=self.audio_service._mic_callback
        )
        mock_mic_stream.start_stream.assert_called_once()
        self.assertFalse(self.audio_service.stop_event.is_set())
        self.assertEqual(self.audio_service.mic_stream, mock_mic_stream)
        self.assertIsNone(self.audio_service.system_stream) # No system audio stream

    def test_start_recording_mic_and_system_audio_success_different_devices(self):
        mock_mic_stream = MagicMock(name="MicStream")
        mock_sys_stream = MagicMock(name="SysStream")
        
        # Mic is device 1, default input (system) is device 0
        self.mock_pa_instance.get_device_info_by_index.return_value = {'name': 'Selected Mic', 'index': 1}
        # self.mock_pa_instance.get_default_input_device_info already returns index 0 from setUp

        self.mock_pa_instance.open.side_effect = [mock_mic_stream, mock_sys_stream]

        result = self.audio_service.start_recording(samplerate=16000, mic_device_index=1, include_system_audio=True)
        self.assertTrue(result)
        
        expected_calls = [
            call(format=self.audio_service.py_format, channels=self.audio_service.channels, rate=16000, input=True, input_device_index=1, frames_per_buffer=self.audio_service.frames_per_buffer, stream_callback=self.audio_service._mic_callback),
            call(format=self.audio_service.py_format, channels=self.audio_service.channels, rate=16000, input=True, input_device_index=0, frames_per_buffer=self.audio_service.frames_per_buffer, stream_callback=self.audio_service._system_callback)
        ]
        self.mock_pa_instance.open.assert_has_calls(expected_calls, any_order=False)
        mock_mic_stream.start_stream.assert_called_once()
        mock_sys_stream.start_stream.assert_called_once()
        self.assertEqual(self.audio_service.mic_stream, mock_mic_stream)
        self.assertEqual(self.audio_service.system_stream, mock_sys_stream)

    def test_start_recording_mic_and_system_audio_same_device(self):
        mock_mic_stream = MagicMock(name="MicStream")
        # Default input (system) is device 0. User explicitly selects device 0 for mic.
        self.mock_pa_instance.get_device_info_by_index.return_value = {'name': 'Selected Mic (Same as Default)', 'index': 0}
        # self.mock_pa_instance.get_default_input_device_info already returns index 0 from setUp

        self.mock_pa_instance.open.return_value = mock_mic_stream # Only one stream should be opened

        result = self.audio_service.start_recording(samplerate=16000, mic_device_index=0, include_system_audio=True)
        self.assertTrue(result)
        self.mock_pa_instance.open.assert_called_once() # Only called for mic stream
        mock_mic_stream.start_stream.assert_called_once()
        self.assertEqual(self.audio_service.mic_stream, mock_mic_stream)
        self.assertIsNone(self.audio_service.system_stream) # System stream should not be opened separately

    def test_start_recording_pyaudio_open_fails(self):
        self.mock_pa_instance.open.side_effect = Exception("PA Error Opening Stream")
        result = self.audio_service.start_recording(samplerate=16000, mic_device_index=0)
        self.assertFalse(result)
        self.mock_pa_instance.open.assert_called_once()
        self.assertIsNone(self.audio_service.mic_stream)

    def test_stop_recording_no_streams_active(self):
        self.audio_service.mic_stream = None
        self.audio_service.system_stream = None
        # Simulate stop_event being set by something else if needed, or just call stop
        self.audio_service.stop_event.clear()

        mixed_audio, rate = self.audio_service.stop_recording()

        self.assertTrue(self.audio_service.stop_event.is_set())
        self.assertIsInstance(mixed_audio, np.ndarray, f"mixed_audio is not a numpy array, it is {type(mixed_audio)}")
        self.assertEqual(mixed_audio.size, 0, f"mixed_audio size is {mixed_audio.size}, expected 0")
        self.assertEqual(rate, self.audio_service.default_samplerate)

    def test_stop_recording_with_active_streams_and_data(self):
        mock_mic_stream = MagicMock(spec=pyaudio.Stream)
        mock_mic_stream.is_active.return_value = True # Simulate active before stop
        self.audio_service.mic_stream = mock_mic_stream

        mock_system_stream = MagicMock(spec=pyaudio.Stream)
        mock_system_stream.is_active.return_value = True
        self.audio_service.system_stream = mock_system_stream

        # Simulate some data being put into chunks by callbacks
        self.audio_service.mic_chunks = [np.random.rand(1024).astype(np.float32) for _ in range(3)]
        self.audio_service.system_chunks = [np.random.rand(1024).astype(np.float32) for _ in range(3)]
        self.audio_service.current_samplerate = 16000

        mixed_audio, rate = self.audio_service.stop_recording()

        self.assertTrue(self.audio_service.stop_event.is_set())
        mock_mic_stream.stop_stream.assert_called_once()
        mock_mic_stream.close.assert_called_once()
        mock_system_stream.stop_stream.assert_called_once()
        mock_system_stream.close.assert_called_once()

        self.assertIsNotNone(mixed_audio)
        self.assertTrue(mixed_audio.size > 0)
        self.assertEqual(rate, 16000)
        # Further checks on mixing logic could be added if np.mean/vstack were directly used here
        # but the mixing happens on concatenated chunks directly in current AudioService code.

    def test_close_terminates_pyaudio(self):
        # setUp already creates self.audio_service which has a self.mock_pa_instance
        self.audio_service.close()
        self.mock_pa_instance.terminate.assert_called_once()
        self.assertIsNone(self.audio_service.pyaudio_instance) # Check it's set to None after termination
        self.assertTrue(self.audio_service._is_terminated)

    def test_callbacks_append_data(self):
        # Test _mic_callback directly (as an example)
        self.audio_service.stop_event.clear()
        self.audio_service.mic_chunks = []
        test_data = np.random.rand(self.audio_service.frames_per_buffer).astype(np.float32).tobytes()
        frame_count = self.audio_service.frames_per_buffer
        time_info = {}
        status_flags = 0
        
        result_mic = self.audio_service._mic_callback(test_data, frame_count, time_info, status_flags)
        self.assertEqual(result_mic, (None, pyaudio.paContinue))
        self.assertEqual(len(self.audio_service.mic_chunks), 1)
        self.assertTrue(np.array_equal(self.audio_service.mic_chunks[0], np.frombuffer(test_data, dtype=np.float32)))

        # Test _system_callback similarly
        self.audio_service.system_chunks = []
        result_sys = self.audio_service._system_callback(test_data, frame_count, time_info, status_flags)
        self.assertEqual(result_sys, (None, pyaudio.paContinue))
        self.assertEqual(len(self.audio_service.system_chunks), 1)
        self.assertTrue(np.array_equal(self.audio_service.system_chunks[0], np.frombuffer(test_data, dtype=np.float32)))

    def test_callback_stops_appending_when_event_set(self):
        self.audio_service.stop_event.set() # Event is set
        self.audio_service.mic_chunks = []
        test_data = b'some_audio_data'
        self.audio_service._mic_callback(test_data, 0, {}, 0)
        self.assertEqual(len(self.audio_service.mic_chunks), 0) # Should not append

if __name__ == '__main__':
    unittest.main() 