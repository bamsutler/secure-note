import io
import pyaudio # Requires PyAudio to be installed
import numpy as np
import threading
import wave # For saving temporary WAV files
import os
from datetime import datetime

from src.config_service import ConfigurationService
from src.logging_service import LoggingService

config_service = ConfigurationService()
log = LoggingService.get_logger(__name__)

class AudioService:
    def __init__(self, config_service: ConfigurationService = ConfigurationService()):
        self.pyaudio_instance = pyaudio.PyAudio()
        self.frames_per_buffer = config_service.get('audio', 'frames_per_buffer', default=1024)
        # Pyaudio format needs to be resolved using getattr from the pyaudio module
        format_str = config_service.get('audio', 'format', default='paFloat32')
        try:
            self.py_format = getattr(pyaudio, format_str)
        except AttributeError:
            log.error(f"Invalid PyAudio format '{format_str}' in config. Using paFloat32 as default.")
            self.py_format = pyaudio.paFloat32
            
        self.channels = config_service.get('audio', 'channels', default=1)
        self.default_samplerate = config_service.get('audio', 'samplerate', default=16000) # Default if not specified in capture
        self.system_audio_keyword = config_service.get('audio', 'system_audio_keyword', default="BlackHole")

        # Recording state
        self.mic_chunks = []
        self.system_chunks = []
        self.stop_event = threading.Event()
        self.mic_stream = None
        self.system_stream = None
        self.current_samplerate = self.default_samplerate
        self._is_terminated = False # Flag to track explicit termination

    def close(self):
        """Explicitly terminate the PyAudio instance and associated resources."""
        if not self._is_terminated and self.pyaudio_instance:
            try:
                # Ensure streams are stopped and closed if open (though stop_recording should handle this)
                if self.mic_stream:
                    if self.mic_stream.is_active(): self.mic_stream.stop_stream()
                    self.mic_stream.close()
                    self.mic_stream = None
                if self.system_stream:
                    if self.system_stream.is_active(): self.system_stream.stop_stream()
                    self.system_stream.close()
                    self.system_stream = None
                
                self.pyaudio_instance.terminate()
                log.info("PyAudio instance explicitly terminated by AudioService.close().")
                self.pyaudio_instance = None # Mark as terminated
                self._is_terminated = True
            except Exception as e:
                log.error(f"Error during explicit PyAudio termination in AudioService.close(): {e}")
                # Still mark as terminated to avoid issues in __del__
                self.pyaudio_instance = None 
                self._is_terminated = True

    def __del__(self):
        """Ensure PyAudio instance is terminated when service object is destroyed, if not already closed."""
        if not self._is_terminated and self.pyaudio_instance:
            # Perform termination silently without logging, as this is a fallback during garbage collection.
            try:
                self.pyaudio_instance.terminate()
            except Exception:
                # Ignore errors during __del__ termination
                pass
            self.pyaudio_instance = None
            self._is_terminated = True

    @staticmethod
    def get_pyaudio_sample_size(format_enum) -> int:
        """Helper to get sample size in bytes for a given PyAudio format enum."""
        # Create a temporary instance just for this static method call, or ensure it's called safely.
        # For simplicity, creating a temporary one here. This is not ideal for frequent calls.
        # A better approach might be to pass the main pyaudio_instance if available, or have a shared utility.
        # However, as it's used in another static method, a temporary instance is common.
        temp_pa = None
        try:
            temp_pa = pyaudio.PyAudio()
            size = temp_pa.get_sample_size(format_enum)
            return size
        except Exception as e:
            log.error(f"Error getting sample size for format {format_enum}: {e}")
            # Fallback or raise. For paInt16, it's 2. Let's make it specific if it fails.
            if format_enum == pyaudio.paInt16:
                log.warning("Falling back to default sample size of 2 for paInt16.")
                return 2
            raise # Re-raise if not a known fallback
        finally:
            if temp_pa:
                temp_pa.terminate()

    def list_input_devices(self) -> list:
        """Lists available audio input devices using PyAudio."""
        devices = []
        try:
            num_devices = self.pyaudio_instance.get_device_count()
            if num_devices == 0:
                log.warning("No audio devices found by PyAudio.")
                return devices

            for i in range(num_devices):
                dev_info = None
                try:
                    dev_info = self.pyaudio_instance.get_device_info_by_index(i)
                    if dev_info.get('maxInputChannels', 0) > 0:
                        host_api_info = self.pyaudio_instance.get_host_api_info_by_index(dev_info.get('hostApi'))
                        devices.append({
                            'id': i, 
                            'name': dev_info.get('name'),
                            'host_api_name': host_api_info.get('name', 'N/A'),
                            'max_input_channels': dev_info.get('maxInputChannels'),
                            'default_sample_rate': dev_info.get('defaultSampleRate')
                        })
                        log.debug(f"Found input device ID {i}: {dev_info.get('name')} (Channels: {dev_info.get('maxInputChannels')})")
                except Exception as e_dev:
                    name = dev_info.get('name') if dev_info else f"Device {i}"
                    log.warning(f"Could not query full details for {name}: {e_dev}")
                    if dev_info and dev_info.get('maxInputChannels', 0) > 0:
                         devices.append({'id': i, 'name': dev_info.get('name', f'Device {i} (partial)')})
            return devices
        except Exception as e:
            log.error(f"Error listing PyAudio devices: {e}")
            return [] # Return empty list on major error

    def get_default_output_device_info(self) -> dict | None:
        """Gets information about the default PyAudio output device."""
        try:
            device_info = self.pyaudio_instance.get_default_output_device_info()
            log.debug(f"Default output device info: {device_info}")
            return device_info
        except Exception as e:
            log.warning(f"Could not get default output device info using PyAudio instance: {e}")
            return None

    def _mic_callback(self, in_data, frame_count, time_info, status_flags):
        if status_flags:
            log.warning(f"PyAudio Mic Callback status flags: {status_flags}")
        if not self.stop_event.is_set():
            self.mic_chunks.append(np.frombuffer(in_data, dtype=np.float32))
        return (None, pyaudio.paContinue)

    def _system_callback(self, in_data, frame_count, time_info, status_flags):
        if status_flags:
            log.warning(f"PyAudio System Callback status flags: {status_flags}")
        if not self.stop_event.is_set():
            self.system_chunks.append(np.frombuffer(in_data, dtype=np.float32))
        return (None, pyaudio.paContinue)

    def start_recording(self, samplerate: int, mic_device_index: int | None = None, include_system_audio: bool = False) -> bool:
        """
        Starts capturing audio from the specified microphone and optionally system audio.
        Args:
            samplerate (int): The sample rate for recording.
            mic_device_index (int | None): Index of the microphone. None for default input.
            include_system_audio (bool): Whether to also capture system audio.
        Returns:
            bool: True if recording started successfully, False otherwise.
        """
        if self.mic_stream or self.system_stream: # Already recording
            log.warning("Recording is already in progress.")
            return False

        self.mic_chunks = []
        self.system_chunks = []
        self.stop_event.clear()
        self.current_samplerate = samplerate
        actual_mic_device_index = mic_device_index

        try:
            # Determine Mic Device Index
            if actual_mic_device_index is None:
                default_mic_info = self.pyaudio_instance.get_default_input_device_info()
                actual_mic_device_index = default_mic_info['index']
                log.info(f"Using default input device for microphone: {default_mic_info['name']} (Index {actual_mic_device_index})")
            else:
                mic_dev_info = self.pyaudio_instance.get_device_info_by_index(actual_mic_device_index)
                log.info(f"Using selected device for microphone: {mic_dev_info['name']} (Index {actual_mic_device_index})")

            self.mic_stream = self.pyaudio_instance.open(
                format=self.py_format, channels=self.channels, rate=samplerate,
                input=True, input_device_index=actual_mic_device_index,
                frames_per_buffer=self.frames_per_buffer, stream_callback=self._mic_callback
            )
            self.mic_stream.start_stream()
            log.info(f"Microphone stream started on device index {actual_mic_device_index} at {samplerate}Hz.")

            if include_system_audio:
                try:
                    system_audio_device_index = None
                    system_device_name = None # To store the name of the device being used for system audio
                    all_input_devices = self.list_input_devices() # Get all devices

                    # Search for the keyword-specified device
                    for device in all_input_devices:
                        device_name_from_list = device.get('name', '')
                        if self.system_audio_keyword.lower() in device_name_from_list.lower():
                            system_audio_device_index = device['id']
                            system_device_name = device_name_from_list
                            log.info(f"Found keyword-matching audio device ('{self.system_audio_keyword}') for system audio: {system_device_name} (Index {system_audio_device_index}).")
                            break
                    
                    if system_audio_device_index is None:
                        log.warning(f"Audio device with keyword '{self.system_audio_keyword}' not found. Falling back to default system input device for system audio.")
                        default_input_info = self.pyaudio_instance.get_default_input_device_info()
                        system_audio_device_index = default_input_info['index']
                        system_device_name = default_input_info['name']
                        log.info(f"Using default input device for system audio: {system_device_name} (Index {system_audio_device_index}).")

                    if system_audio_device_index == actual_mic_device_index:
                        log.warning(f"System audio device ('{system_device_name}') is the same as the selected microphone. System audio will be a duplicate of the microphone audio.")
                        # No need to open a second stream if they are the same device.
                        # Logic in stop_recording will handle duplicating mic_chunks to system_chunks if needed.
                    else:
                        self.system_stream = self.pyaudio_instance.open(
                            format=self.py_format, channels=self.channels, rate=samplerate,
                            input=True, input_device_index=system_audio_device_index,
                            frames_per_buffer=self.frames_per_buffer, stream_callback=self._system_callback
                        )
                        self.system_stream.start_stream()
                        log.info(f"System audio stream started on device: {system_device_name} (Index {system_audio_device_index}) at {samplerate}Hz.")
                except Exception as e_sys:
                    log.error(f"Could not start system audio stream: {e_sys}. System audio capture will be disabled.")
                    self.system_stream = None # Ensure it's None if failed
            return True
        except Exception as e:
            log.exception(f"Error starting PyAudio recording: {e}")
            if self.mic_stream: self.mic_stream.close()
            if self.system_stream: self.system_stream.close()
            self.mic_stream, self.system_stream = None, None
            return False

    def stop_recording(self) -> tuple[np.ndarray | None, int]:
        """
        Stops the current recording, processes and mixes audio data.
        Returns:
            tuple[np.ndarray | None, int]: A tuple containing the mixed audio data (NumPy array)
                                          and the sample rate. Returns (None, rate) if no audio was captured.
        """
        self.stop_event.set()
        samplerate_of_recording = self.current_samplerate

        if self.mic_stream:
            try:
                if self.mic_stream.is_active(): self.mic_stream.stop_stream()
                self.mic_stream.close()
                log.info("Microphone stream stopped and closed.")
            except Exception as e: log.error(f"Error closing mic stream: {e}")
            self.mic_stream = None
        
        if self.system_stream:
            try:
                if self.system_stream.is_active(): self.system_stream.stop_stream()
                self.system_stream.close()
                log.info("System audio stream stopped and closed.")
            except Exception as e: log.error(f"Error closing system stream: {e}")
            self.system_stream = None

        mic_audio_np = np.concatenate(self.mic_chunks, axis=0) if self.mic_chunks else np.array([], dtype=np.float32)
        system_audio_np = np.concatenate(self.system_chunks, axis=0) if self.system_chunks else np.array([], dtype=np.float32)
        
        # If system audio was intended to be a duplicate of mic (same device)
        if not system_audio_np.size and self.mic_chunks and not self.system_stream:
            # This condition implies system_stream wasn't opened because it was same as mic
            log.info("Using microphone audio as system audio (devices were identical or system stream not explicitly opened).")
            system_audio_np = mic_audio_np.copy()

        self.mic_chunks, self.system_chunks = [], [] # Clear for next recording

        if not mic_audio_np.size and not system_audio_np.size:
            log.warning("No audio was recorded from any source.")
            return np.array([], dtype=np.float32), samplerate_of_recording

        mixed_audio_np = None
        if mic_audio_np.size > 0 and system_audio_np.size > 0:
            log.info("Mixing microphone and system audio...")
            len_mic, len_system = len(mic_audio_np), len(system_audio_np)
            max_len = max(len_mic, len_system)
            mic_padded = np.pad(mic_audio_np, (0, max_len - len_mic), 'constant') if len_mic < max_len else mic_audio_np
            system_padded = np.pad(system_audio_np, (0, max_len - len_system), 'constant') if len_system < max_len else system_audio_np
            mixed_audio_np = 0.5 * mic_padded + 0.5 * system_padded
            log.info(f"Mixed audio created. Length: {len(mixed_audio_np)} samples.")
        elif mic_audio_np.size > 0:
            log.info("Using only microphone audio as system audio was not available/captured.")
            mixed_audio_np = mic_audio_np
        elif system_audio_np.size > 0:
            log.info("Using only system audio as microphone was not available/captured.")
            mixed_audio_np = system_audio_np
        
        return mixed_audio_np, samplerate_of_recording

    @staticmethod
    def convert_float32_to_wav_bytes(audio_data_float32: np.ndarray, samplerate: int, channels: int = 1) -> bytes | None:
        """Converts float32 NumPy audio data to WAV byte stream."""
        if audio_data_float32 is None or audio_data_float32.size == 0:
            log.warning("Cannot convert empty audio data to WAV bytes.")
            return None
        try:
            # Normalize to 16-bit range and convert to int16
            audio_data_int16 = (audio_data_float32 * 32767).astype(np.int16)
            
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(channels)
                wf.setsampwidth(AudioService.get_pyaudio_sample_size(pyaudio.paInt16)) # 2 bytes for paInt16
                wf.setframerate(samplerate)
                wf.writeframes(audio_data_int16.tobytes())
            wav_bytes = buffer.getvalue() # Get value from the BytesIO buffer after wave is closed
            log.debug(f"Successfully converted float32 audio to WAV bytes (length: {len(wav_bytes)}).")
            return wav_bytes
        except Exception as e:
            log.exception(f"Error converting float32 audio to WAV bytes: {e}")
            return None

