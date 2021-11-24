import pyaudio

CHUNK_LENGTH = 1000
FORMAT: int = pyaudio.paInt16
N_CHANNELS: int = 1
RATE: int = 10000
TEMPORAL_VLIM: float = 35000.0
WAVEFORM_VLIM: float = 1e6
