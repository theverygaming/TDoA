import fractions
import numpy as np
import scipy
import matplotlib.pyplot as plt
import tools
import tdoa

def spectrogram(x, size, fs):
    plt.figure(figsize=(12, 4))
    plt.specgram(x, NFFT=size, Fs=fs, scale="dB", vmin=-100, cmap="viridis")
    plt.colorbar(label="Spectral Density (dB)")
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Frequency (Hz)")

def plot(x, fs):
    plt.figure(figsize=(12, 4))
    t = np.arange(len(x)) / fs
    plt.plot(t, x)
    plt.xlabel("Time (Seconds)")
    plt.ylabel("Voltage")

def to_wav(file, x, fs):
    x = normalize(x)
    if np.iscomplexobj(x):
        x = np.column_stack((np.real(x), np.imag(x)))
    scipy.io.wavfile.write(file, fs, (np.clip(x * 32767, -32768, 32767)).astype(np.int16).reshape(-1, 2))


def normalize(x):
    return x / np.max(np.abs(x))

def carrier(t, f):
    return np.exp(1j * 2 * np.pi * f * t)

def am(c, m, mod_index):
    return normalize((1 + (mod_index * normalize(m))) * c)

def fm(t, fc, m, kf):
    dt = t[1] - t[0]
    phase = 2 * np.pi * fc * t + 2 * np.pi * kf * np.cumsum(normalize(m)) * dt
    return np.exp(1j * phase)

def pm(t, fc, m):
    return np.exp(1j * (2 * np.pi * fc * t + m))

def mod_data(sps, data, symmap):
    symmap = {
        0: -1,
        1: 1,
    }
    syms = sum([[symmap[(bt >> bi) & 1] for bi in range(8)] for bt in data], start=[])
    syms_interp = np.ndarray(len(syms)*sps, dtype=np.float32)
    for i, x in enumerate(syms):
        syms_interp[i*sps:(i*sps)+sps] = x

    return syms_interp

def noise(t):
    return (
        (np.random.uniform(-1, 1, len(t)) * 1j)
        + np.random.uniform(-1, 1, len(t))
    )

def delay(x, delay_samples):
    return np.concatenate((np.zeros(delay_samples, dtype=x.dtype), x[:len(x) - delay_samples]))

def mix(t, x, f):
    c = carrier(t, f)
    if not np.iscomplexobj(x):
        c = np.real(c)
    return x * carrier(t, f)

def vfo(t, x, fs, fcenter, bw):
    x = mix(t, x, -fcenter)
    sos = scipy.signal.butter(10, bw/2, "lowpass", fs=fs, output="sos")
    x = scipy.signal.sosfilt(sos, x)
    return x

def hf_path(t, x, delay_samples, fade_strength=1, phase_slow=0, phase_mid=0, f_slow=0.03, f_mid=0.3):
    # multi-second (usually at least 10) slow up and down fading
    slow = np.cos(2 * np.pi * f_slow * t + phase_slow)
    # faster fading
    mid = np.cos(2 * np.pi * f_mid * t + phase_mid) * 2
    # some noise
    fast = np.real(noise(t)) * 0.01

    envelope = slow + mid + fast
    envelope = envelope - envelope.min()
    envelope = envelope / envelope.max()
    envelope = (1 - fade_strength) + fade_strength * envelope
    return normalize(delay(x, delay_samples) * envelope)

def gen_am(f_carrier, fs, len_s):
    t = np.arange(0, len_s, 1 / fs)
    m = (
        np.real(carrier(t, 1000))
        + np.real(fm(t, 2500, np.real(carrier(t, 1)), 500))
    )
    sig = am(carrier(t, f_carrier), m, 1)
    return t, sig

def gen_fsk(f_carrier, shift, baudrate, fs, len_s):
    t = np.arange(0, len_s, 1 / fs)
    sig = fm(
        t,
        f_carrier,
        np.pad(
            (md := mod_data(int(fs*(1/baudrate)), b"The quick brown fox jumps over the lazy dog :3" + bytes(np.zeros(1)), {0: -1, 1: -1})[:len(t)]),
            (0, len(t) - len(md)),
            mode="wrap",
        ),
        shift/2,
    )
    return t, sig

def gen_tdoa_recs(
    t,
    signal,
    signal_pbw,
    signal_fcenter,
    fs,
    tx_position: tuple[float, float],
    rx_params: list[str, tuple[tuple[float, float]]],
    out_fs = 12000,
) -> list[tdoa.TDoAPositionedRecording]:
    # FIXME: don't calculate the delay twice lol
    max_delay_samps = max([round(fs * (tools.haversine(*tx_position, *rx[1]) / scipy.constants.c)) for rx in rx_params])
    print(f"max delay: {max_delay_samps}")
    # TODO: random fading parameters
    def gen_rec(rxname, rx_position, noise_db):
        dist_m = tools.haversine(*tx_position, *rx_position)
        delay_samps = round(fs * (dist_m / scipy.constants.c))
        print(f"delay: {delay_samps}")

        s = signal
        s = hf_path(t, s, delay_samps, fade_strength=0)
        s = normalize(s)

        # remove max. delay from start because the delay messes up the start of the signal
        tn = t[max_delay_samps:]
        s = s[max_delay_samps:]

        # Add noise
        s += noise(tn) * (10 ** (noise_db / 10))
        s = normalize(s)

        # tune the signal aroud 0
        s = normalize(vfo(tn, s, fs, signal_fcenter, signal_pbw))

        # resample
        frac = fractions.Fraction(out_fs / fs).limit_denominator(20)
        interpolation = frac.numerator
        decimation = frac.denominator
        s = scipy.signal.resample_poly(s, interpolation, decimation)
        fs_rs = (fs * interpolation) / decimation

        # generate timestamps
        samp_dt_ns = int((1 / fs_rs) * 1e9)
        timestamps = int(1500000000 * 1e9) + np.arange(len(s), dtype=np.int64) * samp_dt_ns

        return tdoa.TDoAPositionedRecording(
            s,
            timestamps,
            fs_rs,
            *rx_position,
            rxname,
        )
    return [gen_rec(*rx) for rx in rx_params]
