import librosa
import numpy as np
import soundfile as sf
from numpy.typing import NDArray
from scipy.signal import istft, stft

from algo.beamformer import apply_beamformer_stft, wng_mvdr_steepest
from algo.noise_estimation import est_Rnn, reduce_Rnn_pca
from util.configure import Config
from util.simulate import Mic_Type, sim_mic, sim_room
from util.visualize import plot_mic_pos, plot_room_pos

# Define configuration
config = Config()

# Simulate room
room = sim_room(config.room_dim.tolist(), config.fs, config.reflection_count)

# Simulate microphone array
mic, mic_pos = sim_mic(
    config.mic_count,
    config.mic_loc,
    config.mic_spacing,
    getattr(Mic_Type, config.mic_type.upper()),
    config.fs,
)

# Add microphone array to room
room.add_microphone_array(mic)  # type: ignore[reportUnknownMemberType]

# Define signal sources
signal_sources = [source for source in config.sources if source.classification == "signal"]

# Verify only one signal is defined
if len(signal_sources) != 1:
    # Print error message
    msg = f"Expected only 1 signal source instead of {len(signal_sources)}"
    raise ValueError(msg)

else:
    # Extract signal location
    signal_loc = signal_sources[0].loc


# Iterate through sources
for source in config.sources:
    # Load audio source
    data, fs = librosa.load(source.input, sr=config.fs)

    # Verify sampling rate is correct
    if fs != config.fs:
        # Print error message
        msg = "Room and audio source sampling rates do not match"
        raise ValueError(msg)

    # Add audio source to room
    room.add_source(source.loc, signal=data)  # type: ignore[reportUnknownMemberType]

# Visualize microphone and room layouts
plot_mic_pos(mic_pos, config.output_dir)
plot_room_pos(config.room_dim, config.mic_loc, config.sources, config.output_dir)

# Run simulation
room.simulate()  # type: ignore[reportUnknownMemberType]

# Record simulated microphone signals
mic_audio: NDArray[np.float64] = np.array(room.mic_array.signals)  # type: ignore[reportUnknownMemberType]

# Create image directory if it does not exist
audio_dir = config.output_dir / "audio"
if not audio_dir.exists():
    audio_dir.mkdir(parents=True)

# Write recorded audio to wavefile
sf.write(audio_dir / "mic_raw_audio.wav", mic_audio.T, config.fs)  # type: ignore[reportUnknownMemberType]

# Determine audio dimensions
sample_count, mic_count = mic_audio.T.shape

# Verify audio dimensions
if mic_count != config.mic_count:
    # Print error message
    msg = "Microphone count of recorded audio does not match configuration"
    raise ValueError(msg)

# Select noise-only segment from microphone audio, future work will implement VAD
noise_start_time = 20.0
noise_end_time = 30.0
noise_start_idx = int(noise_start_time * config.fs)
noise_end_idx = int(noise_end_time * config.fs)
mic_noise = mic_audio.T[noise_start_idx:noise_end_idx, :]

# Compute noise covariance for noise-only segment
Rnn = est_Rnn(mic_noise)

# Apply principal component analysis
if config.pc_count > 0:
    Rnn = reduce_Rnn_pca(Rnn, config.pc_count)

# Choose STFT window size
Nfft = int(config.fs * config.frame_duration / 1000)

# Use 50% overlap with Hann window
hop = Nfft // 2
window = np.hanning(Nfft)

# Initialize list to store the STFT matrices of each channel
stft_list: list[NDArray[np.complex128]] = list()
fvec: NDArray[np.float64] = np.empty(0, dtype=np.float64)

# Iterate over microphone channels
for mic_idx in range(config.mic_count):
    # Extract time-domain signal for one microphone channel
    mic_signal: NDArray[np.float64] = mic_audio[:, mic_idx]

    # Compute STFT
    fvec, _, stft_matrix = stft(  # type: ignore[reportUnknownMemberType]
        mic_audio.T[:, mic_idx],
        fs=config.fs,
        nperseg=Nfft,
        noverlap=Nfft - hop,
        window=window,  # type: ignore[reportUnknownMemberType]
        padded=True,
        return_onesided=True,
    )

    # Cast matrix type to ensure compatibility
    stft_matrix = np.asarray(stft_matrix, dtype=np.complex128)

    # Append STFT of microphone channel
    stft_list.append(stft_matrix)

# Convert list to array, all channel samples from time step grouped into same array
stft_output = np.stack(stft_list, axis=-1)

# Extract dimensional information
freq_bin_count, time_frame_count, _ = stft_output.shape

# Compute distance between microphones and signal source
dist = np.linalg.norm(mic_pos.T - (signal_loc - config.mic_loc), axis=1)

# Determine time delays per microphone from signal source
tau = dist / config.sound_speed

# Initialize steering matrix
steering_vec = np.zeros((freq_bin_count, config.mic_count), dtype=complex)

# Verify frequency vector is defined
if fvec.size == 0:  # type: ignore[reportUnknownMemberType]
    msg = "No STFT was computed"
    raise ValueError(msg)

# Iterate over frequency vectors per microphone
for freq_idx, freq in enumerate(fvec):  # type: ignore[reportUnknownMemberType]
    # Determine phase delays
    steering_vec[freq_idx, :] = np.exp(-1j * 2 * np.pi * freq * tau)

# Define optimization parameters
gamma_dB = 15
gamma = 10 ** (gamma_dB / 10)
mu = 0.01
Kiter = 200

# Compute WNG-MVDR weights for each frequency bin with steepest descent method
weights_steepest = np.zeros((freq_bin_count, config.mic_count), dtype=complex)
for kf in range(freq_bin_count):
    a = steering_vec[kf, :].reshape(-1, 1)
    w = wng_mvdr_steepest(Rnn, a, gamma, mu, Kiter)
    weights_steepest[kf, :] = w[:, 0]

# Compute WNG-MVDR weights for each frequency bin with Newton's method
weights_newton = np.zeros((freq_bin_count, config.mic_count), dtype=complex)
for kf in range(freq_bin_count):
    a = steering_vec[kf, :].reshape(-1, 1)
    w = wng_mvdr_steepest(Rnn, a, gamma, mu, Kiter)
    weights_newton[kf, :] = w[:, 0]

# Apply beamformer
freq_steepest = apply_beamformer_stft(stft_output, weights_steepest)
freq_newton = apply_beamformer_stft(stft_output, weights_newton)

# Apply Inverse STFT
_, time_steepest = istft(  # type: ignore[reportUnknownMemberType]
    freq_steepest,
    fs=config.fs,
    nperseg=Nfft,
    noverlap=Nfft - hop,
    window=window,  # type: ignore[reportUnknownMemberType]
)
_, time_newton = istft(  # type: ignore[reportUnknownMemberType]
    freq_newton,
    fs=config.fs,
    nperseg=Nfft,
    noverlap=Nfft - hop,
    window=window,  # type: ignore[reportUnknownMemberType]
)

# Ensure audio is real before writing to wav file
time_steepest = np.real(time_steepest)
time_newton = np.real(time_newton)

# Match original audio length
if len(time_steepest) > sample_count:
    time_steepest = time_steepest[:sample_count]
else:
    time_steepest = np.pad(time_steepest, (0, sample_count - len(time_steepest)))
if len(time_newton) > sample_count:
    time_newton = time_newton[:sample_count]
else:
    time_newton = np.pad(time_newton, (0, sample_count - len(time_newton)))

# Write filtered audio to wavefiles
sf.write(audio_dir / "mic_steepest_filtered_audio.wav", time_steepest, config.fs)  # type: ignore[reportUnknownMemberType]
sf.write(audio_dir / "mic_newton_filtered_audio.wav", time_newton, config.fs)  # type: ignore[reportUnknownMemberType]

# Compare optimization methods
