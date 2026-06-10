import dataclasses
import itertools
import numpy as np
import numpy.typing as npt
import scipy
import tools


@dataclasses.dataclass
class TDoARecording:
    # IQ samples
    samples: npt.NDArray[np.complex64]
    # nanosecond timestamps for each sample in samples
    timestamps: npt.NDArray[np.int64]
    # sample rate
    sr: float # TODO: maybe compute this from timestamps and cache it?

    @staticmethod
    def sync_recs(recs: list["TDoARecording"], time_diff_max_ns=1e6, time_diff_warn_ns=1e5):
        """
        Throws away all parts of the specified recordings that are not overlapping.

        Operates on references of the recordings!
        """

        # ensure all recs start at around the same time
        latest_start = max(rec.timestamps[0] for rec in recs)

        sr_median = np.median([rec.sr for rec in recs])

        i = 0
        while i < len(recs):
            if abs(sr_median - recs[i].sr) > (sr_median / 100):
                print(f"dropping recording {recs[i]} due to unusual SR {recs[i].sr} (median {sr_median})")
                del recs[i]
                continue
            t_end = recs[i].timestamps[-1]
            if t_end < latest_start:
                print(f"dropping recording {recs[i]} due to end before lastest_start")
                del recs[i]
                continue
                #raise Exception(f"rec ({rec}) ends before latest_start")
            i += 1

        for i, rec in enumerate(recs):
            diff = np.absolute(rec.timestamps - latest_start)
            min_idx = diff.argmin()
            if diff[min_idx] > time_diff_max_ns:
                raise Exception(f"recs ({rec}) are desynced by more than time_diff_max_ns ({time_diff_max_ns} ns) from each other ({diff[min_idx]} ns)")
            if diff[min_idx] > time_diff_warn_ns:
                print(f"WARNING: sync_recs: diff {diff[min_idx]} ns ({((diff[min_idx] / 1e9) * scipy.constants.c) / 1000} km)")
            recs[i].timestamps = rec.timestamps[min_idx:]
            recs[i].samples = rec.samples[min_idx:]

        # ensure all recs have the same length
        smallest_len = min(len(rec.timestamps) for rec in recs)
        for i, rec in enumerate(recs):
            recs[i].timestamps = rec.timestamps[:smallest_len]
            recs[i].samples = rec.samples[:smallest_len]

    def split(self, max_secs):
        max_dt_ns = int(max_secs * 1e9)
        samples = self.samples
        timestamps = self.timestamps
        splits = []
        while True:
            (indicies,) = np.asarray(timestamps >= timestamps[0] + max_dt_ns).nonzero()
            if len(indicies) == 0:
                splits.append(dataclasses.replace(self, samples=samples, timestamps=timestamps))
                break
            idx = indicies[0]
            splits.append(dataclasses.replace(self, samples=samples[:idx], timestamps=timestamps[:idx]))
            samples = samples[idx:]
            timestamps = timestamps[idx:]
        return splits

    def cut(self, start: float, end: float):
        """
        Cut a recording to a timeframe from start to end (in seconds)
        """
        s_start = int(self.sr * start)
        s_end = int(self.sr * end)
        self.samples = self.samples[s_start:s_end]
        self.timestamps = self.timestamps[s_start:s_end]

    def resample(self, up, down):
        ratio = up / down
        self.sr *= ratio

        self.samples = scipy.signal.resample_poly(self.samples, up, down)
        self.timestamps = np.interp(
            np.arange(len(self.samples), dtype=np.int64),
            np.arange(len(self.timestamps), dtype=np.int64),
            self.timestamps,
        )

    def plot_spectrogram(self, fig, ax):
        _, _, _, im = ax.specgram(self.samples, NFFT=int(self.sr / 10), Fs=self.sr, scale="dB", vmin=-100, cmap="viridis")
        fig.colorbar(im, ax=ax, orientation="vertical", label="Spectral Density (dBFS)")
        ax.set_xlabel("Time (Seconds)")
        ax.set_ylabel("Frequency (Hz)")

    @classmethod
    def stitch(cls, recs: list["TDoARecording"], noise=True):
        """
        Stitch multiple recordings together into one, fill spaces with null data
        """
        sr_median = np.median([rec.sr for rec in recs])
        for rec in recs:
            if abs(sr_median - rec.sr) > (sr_median / 100):
                raise Exception(f"stitch: recording {rec} has out of line sample rate {rec.sr}")

        recs_sorted = sorted(recs, key=lambda x: x.timestamps[0])

        sample_dt_ns = int((1 / sr_median) * 1e9)
        samples = []
        timestamps = []
        ts_last_end = recs_sorted[0].timestamps[0] - sample_dt_ns
        for rec in recs_sorted:
            # fill in gap with zeros if needed
            dt = rec.timestamps[0] - ts_last_end
            if dt < 0:
                raise Exception(f"stitch: recordings overlap")
            space_samples = int(round(dt / sample_dt_ns))
            if space_samples == 0:
                raise Exception(f"stitch: there is less than one sample of spacing betwen this and the previous recording dt: {dt} sample_dt_ns: {sample_dt_ns}")
            if space_samples > 1:
                space_samples -= 1
                if noise:
                    samples.append(np.random.uniform(-1e-12, 1e-12, space_samples * 2).astype(np.float32).view(np.complex64))
                else:
                    samples.append(np.zeros(space_samples, dtype=np.complex64))
                timestamps.append(ts_last_end + sample_dt_ns + (np.arange(space_samples, dtype=np.int64) * sample_dt_ns))
            samples.append(rec.samples)
            timestamps.append(rec.timestamps)
            ts_last_end = rec.timestamps[-1]

        samples = np.concatenate(samples)
        timestamps = np.concatenate(timestamps)
        assert len(samples) == len(timestamps)

        return cls(
            samples=samples,
            timestamps=timestamps,
            sr=sr_median,
            **{k: v for k, v in dataclasses.asdict(recs_sorted[0]).items() if k not in ["samples", "timestamps", "sr"]},
        )


@dataclasses.dataclass
class TDoAPositionedRecording(TDoARecording):
    lat: float
    lon: float
    name: str | None = None

    @classmethod
    def from_recording(cls, rec: TDoARecording, lat: float, lon: float, name: str | None = None):
        return cls(
            samples=rec.samples,
            timestamps=rec.timestamps,
            sr=rec.sr,
            lat=lat,
            lon=lon,
            name=name,
        )


class TDoAAlgorithm:
    # TODO: docstrings?

    def get_dist_intensity_fn(self, r1: TDoARecording, r2: TDoARecording):
        raise NotImplementedError()


class TDoAAlgorithmSimple(TDoAAlgorithm):
    def __init__(self, max_dist_m=10000*1000, demod=None):
        self._max_dist_m = max_dist_m

        match (demod.lower() if demod is not None else demod):
            case "phase":
                self._demod = lambda x, _: np.angle(x) / (2 * np.pi)
            case "fm":
                def demod_fm(sig, bandwidth, samplerate):
                    # https://github.com/AlexandreRouma/SDRPlusPlus/blob/36ea9a143422f5b374371461667ff53fb9387300/core/src/dsp/demod/quadrature.h
                    inv_deviation = 2 * np.pi * ((bandwidth / 2) / samplerate)
                    phase = np.angle(sig) # np.angle is equal to np.arctan2(im, re)
                    demod = np.diff(np.unwrap(phase) / inv_deviation)
                    return np.pad(demod, (1, 0), mode="edge")
                self._demod = lambda x, _: demod_fm(x, 1, 1)
            case "am":
                def demod_am(x, sr):
                    # get magnitude and remove DC (AM demod)
                    x1 = np.abs(x)
                    sos = scipy.signal.butter(4, 0.1, "hp", fs=sr, output="sos")
                    x1 = scipy.signal.sosfiltfilt(sos, x1)
                    return x1
                self._demod = demod_am
            case None:
                self._demod = lambda x, _: x
            case _:
                raise Exception(f"unknown demod '{demod}'")

    def get_dist_intensity_fn(self, r1: TDoARecording, r2: TDoARecording):
        lag_time, intensity = self._compute_recording_lags(r1, r2)
        def get_corr():
            return lag_time, intensity
        # convert seconds lag to distance in m
        lag_dist = lag_time * scipy.constants.c
        # TODO: maybe do some magic to obtain a more accurate measurement even when the resulution is bad?
        peak_dist = lag_dist[np.argmax(intensity)]
        return self._get_dist_intensity_fn(lag_dist, intensity), get_corr, peak_dist, self._rate_corr(intensity)

    def _compute_lags(self, s1, s2, sr, max_dist_m):
        # this is in essence similar to
        # https://github.com/hcab14/TDoA/blob/2bb9dc2ecc2c6ebcc13ed11c7cbeadea0cd5dfcd/m/tdoa_compute_lags_new.m#L20-L28
        # and ofc strongly inspired by that

        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12, 4))
        # t = np.arange(len(s2)) / sr
        # plt.plot(t, s2)
        # plt.xlabel("Time (Seconds)")
        # plt.ylabel("Voltage A")

        # demodulate
        s1 = self._demod(s1, sr)
        s2 = self._demod(s2, sr)

        # remove any constant DC offsets
        s1 -= np.mean(s1)
        s2 -= np.mean(s2)

        # plt.figure(figsize=(12, 4))
        # t = np.arange(len(s2)) / sr
        # plt.plot(t, s2)
        # plt.xlabel("Time (Seconds)")
        # plt.ylabel("Voltage B")
        # plt.show()

        corr = scipy.signal.correlate(s1, s2, mode="full")

        # lag indices
        lags = scipy.signal.correlation_lags(len(s1), len(s2), mode="full")

        # normalize correlation
        corr = corr / (np.sqrt(np.sum(np.abs(s1) ** 2) * np.sum(np.abs(s2) ** 2) + 1e-12))

        # Distance limit; This also makes things a tiny bit faster as we work with less data
        if max_dist_m is not None:
            max_lag = int((max_dist_m / scipy.constants.c) * sr)
            center = len(corr) // 2
            window = slice(center - max_lag, center + max_lag + 1)
            corr = corr[window]
            lags = lags[window]

        # convert lags to seconds
        lag_time = lags / sr

        intensity = np.abs(corr)

        return lag_time, intensity

    def _compute_recording_lags(self, r1, r2):
        start_offset = (r1.timestamps[0] - r2.timestamps[0]) / 1e9
        lag_time, intensity = self._compute_lags(
            r1.samples,
            r2.samples,
            np.mean([r1.sr, r2.sr]),
            self._max_dist_m,
        )
        lag_time += start_offset
        return lag_time, intensity

    def _get_dist_intensity_fn(self, lag_dist, intensity):
        # function that will, given a distance in meters return the intensity at that point
        return scipy.interpolate.interp1d(
            lag_dist,
            intensity,
            bounds_error=False,
            fill_value=np.min(intensity),
        )

    def _rate_corr(self, correlation):
        peaks, peak_properties = scipy.signal.find_peaks(correlation, height=0, prominence=0, width=0)
        if len(peaks) == 0:
            return 0

        # identify main peak
        main_peak_idx = np.argmax(peak_properties["peak_heights"])
        main_peak_prominence = peak_properties["prominences"][main_peak_idx]
        main_peak_width = max(peak_properties["widths"][main_peak_idx], 1) / len(correlation)

        # find peaks at least a quarter as tall as the main one
        height_threshold = peak_properties["peak_heights"][main_peak_idx] / 4
        n_peaks = np.sum(peak_properties["peak_heights"] >= height_threshold)

        score = float(np.round(main_peak_prominence / (main_peak_width * n_peaks), 4))

        return score


class TDoARun:
    def __init__(self, algorithm: TDoAAlgorithm, recs: list[TDoAPositionedRecording], ref_rec_idx: None | int, p1, p2):
        self._algorithm = algorithm
        self._recs = recs
        self._ref_rec_idx = ref_rec_idx
        self._rx_dist_fns = {}
        self._heatmap_cache = {}

        if len(self._recs) < 2:
            raise Exception(f"need at least two recordings for TDoA, got {len(self._recs)}")

        # for r in self._recs:
        #     r.resample(10, 1)

        # calculate needed resolution in degrees to represent max resolution
        distance_per_degree_lat = (2 * np.pi * tools.EARTH_RADIUS_M) / 360
        res_m = self.get_max_res()
        lat_min = min([p1[0], p2[0]])
        lat_max = max([p1[0], p2[0]])
        # equator cross?
        if lat_min <= 0 and 0 <= lat_max:
            max_res_lat = 0
        else:
            max_res_lat = lat_min if abs(lat_min) < abs(lat_max) else lat_max
        res_deg = res_m / (distance_per_degree_lat * np.cos(np.radians(max_res_lat)))
        # beautify the number a litle
        res_deg = tools.dynamic_round(res_deg, 3)
        print(f"heatmap resolution in degrees {res_deg} to represent max resolution in meters {res_m}")

        self._latgr, self._longr, self._intensity_template = self._prepare_heatmap(p1, p2, res_deg)

        TDoARecording.sync_recs(self._recs)

    @staticmethod
    def _prepare_heatmap(p1, p2, res):
        lat_min = min(p1[0], p2[0])
        lat_max = max(p1[0], p2[0])
        lon_min = min(p1[1], p2[1])
        lon_max = max(p1[1], p2[1])

        lats = np.arange(lat_min, lat_max, res)
        lons = np.arange(lon_min, lon_max, res)

        longr, latgr = np.meshgrid(lons, lats)

        intensity = np.zeros_like(latgr, dtype=np.float32)

        return latgr, longr, intensity

    def _compute_rec_dists(self):
        if self._rx_dist_fns:
            return

        if self._ref_rec_idx is not None:
            for i in range(len(self._recs)):
                if i == self._ref_rec_idx:
                    continue
                self._rx_dist_fns[(self._ref_rec_idx, i)] = self._algorithm.get_dist_intensity_fn(self._recs[self._ref_rec_idx], self._recs[i])
                print(f"computed corr for {i+1}/{len(self._recs)}")
        else:
            combos = list(itertools.combinations(range(len(self._recs)), 2))
            for i, (a, b) in enumerate(combos):
                self._rx_dist_fns[(a, b)] = self._algorithm.get_dist_intensity_fn(self._recs[a], self._recs[b])
                print(f"computed corr for {i+1}/{len(combos)}")

    def _get_heatmap(self, recpairs: list[tuple[int, int]]):
        intensity = np.copy(self._intensity_template)
        pair_intensities = []
        for i, (a, b) in enumerate(recpairs):
            if (a, b) not in self._heatmap_cache:
                d1 = tools.haversine(self._latgr, self._longr, self._recs[a].lat, self._recs[a].lon)
                d2 = tools.haversine(self._latgr, self._longr, self._recs[b].lat, self._recs[b].lon)

                dist = d1 - d2

                self._heatmap_cache[(a, b)] = self._rx_dist_fns[(a, b)][0](dist)
                print(f"cached heatmap {i+1}/{len(recpairs)}")

            pair_intensities.append((self._rx_dist_fns[(a, b)][3], self._heatmap_cache[(a, b)], f"{self._recs[a].name} - {self._recs[b].name}"))

        # pair_intensities.sort(key=lambda x: x[0], reverse=True)
        # pair_intensities = pair_intensities[:200]

        for rating, it, name in pair_intensities:
            intensity += it

        intensity /= len(pair_intensities)

        return intensity

    def get_pairs(self):
        self._compute_rec_dists()
        intensities = {}
        for recids in self._rx_dist_fns:
            intensities[recids] = self._get_heatmap([recids])
        return intensities

    def get_all(self):
        self._compute_rec_dists()

        intensity = self._get_heatmap(self._rx_dist_fns.keys())

        # rate recordings
        # TODO: move somewhere else maybe lol
        rec_ratings = {}
        for (a, b) in self._rx_dist_fns:
            for x in [a, b]:
                if x not in rec_ratings:
                    rec_ratings[x] = []
            rating = self._rx_dist_fns[(a, b)][3]
            rec_ratings[a].append(rating)
            rec_ratings[b].append(rating)
        for k in rec_ratings:
            rec_ratings[k] = np.mean(rec_ratings[k])
        for name, rating in sorted([(self._recs[k].name, s) for k, s in rec_ratings.items()], key=lambda x: x[1], reverse=True):
            print(f"rx '{name}' rating {rating}")

        return intensity

    def get_rec(self, recid):
        return self._recs[recid]

    def get_grid(self):
        return self._latgr, self._longr

    def get_max_res(self):
        """
        Returns the Maximum possible resolution in meters
        """
        sr = np.mean([r.sr for r in self._recs])
        return (1 / sr) * scipy.constants.c

    def plot_correlation(self, fig, ax, rxid1, rxid2):
        self._compute_rec_dists()
        lag_time, intensity = self._rx_dist_fns[(rxid1, rxid2)][1]()
        ax.plot(lag_time, intensity)
        ax.set_xlabel("Lag (Seconds)")
        ax.set_ylabel("Intensity")
        ax.set_ylim([0, 1])
