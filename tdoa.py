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
    sr: float

    @staticmethod
    def sync_recs(recs: list["TDoARecording"], time_diff_max_ns=2e9):
        """
        Throws away all parts of the specified recordings that are not overlapping.

        Operates on references of the recordings!
        """
        # TODO: throw some errors when a recording is completely out of line and there are no overlaps? Or are we doing that already maybe?

        # ensure all recs start at around the same time
        latest_start = max(rec.timestamps[0] for rec in recs)
        for i, rec in enumerate(recs):
            diff = np.absolute(rec.timestamps - latest_start)
            start_idx = diff.argmin()
            if diff[start_idx] > time_diff_max_ns:
                raise Exception(f"recs start more than time_diff_max_ns ({time_diff_max_ns}) apart from each other")
            recs[i].timestamps = rec.timestamps[start_idx:]
            recs[i].samples = rec.samples[start_idx:]

        # ensure all recs have the same length
        smallest_len = min(len(rec.timestamps) for rec in recs)
        for i, rec in enumerate(recs):
            recs[i].timestamps = rec.timestamps[:smallest_len]
            recs[i].samples = rec.samples[:smallest_len]


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


class TDoARun:
    def __init__(self, recs: list[TDoAPositionedRecording], p1, p2, res):
        self._recs = recs
        self._rx_lags = {}
        self._latgr, self._longr, self._m = self._prepare_heatmap(p1, p2, res)

        if len(self._recs) < 2:
            raise Exception(f"need at least two recordings for TDoA, got {len(self._recs)}")

        TDoARecording.sync_recs(self._recs)

    @staticmethod
    def _compute_lags(s1, s2, sr, max_dist_m=10000*1000):
        # this is in essence similar to
        # https://github.com/hcab14/TDoA/blob/2bb9dc2ecc2c6ebcc13ed11c7cbeadea0cd5dfcd/m/tdoa_compute_lags_new.m#L20-L28
        # and ofc strongly inspired by that

        corr = scipy.signal.correlate(s1, s2, mode="full")

        # lag indices
        lags = scipy.signal.correlation_lags(len(s1), len(s2), mode="full")

        # Distance limit; This also makes things a tiny bit faster as we work with less data
        if max_dist_m is not None:
            max_lag = int((max_dist_m / scipy.constants.c) * sr)
            center = len(corr) // 2
            window = slice(center - max_lag, center + max_lag + 1)
            corr = corr[window]
            lags = lags[window]

        # convert lags to seconds
        lag_time = lags / sr

        # normalize correlation
        corr = corr / np.max(np.abs(corr))

        # probability magic? idfk lmfaosob, kiwisdr TDoA this
        sigma2 = -2 * np.log(np.abs(corr))

        return lag_time, sigma2

    @classmethod
    def _compute_recording_lags(cls, r1, r2):
        start_offset = (r1.timestamps[0] - r2.timestamps[0]) / 1e9
        lag_time, sigma2 = cls._compute_lags(
            r1.samples,
            r2.samples,
            np.mean([r1.sr, r2.sr]),
        )
        lag_time += start_offset
        return lag_time, sigma2

    @staticmethod
    def _get_dist_probability_fn(lag_time, sigma2):
        # convert seconds lag to distance in m
        lag_time *= scipy.constants.c
        # function that will, given a distance in meters return the sigma2 at that point
        probability_func = scipy.interpolate.interp1d(
            lag_time,
            sigma2,
            bounds_error=False,
            fill_value=np.max(sigma2), # TODO: maybe just inf?
        )
        return probability_func

    @staticmethod
    def _prepare_heatmap(p1, p2, res):
        lat_min = min(p1[0], p2[0])
        lat_max = max(p1[0], p2[0])
        lon_min = min(p1[1], p2[1])
        lon_max = max(p1[1], p2[1])

        lats = np.arange(lat_min, lat_max, res)
        lons = np.arange(lon_min, lon_max, res)

        longr, latgr = np.meshgrid(lons, lats)

        # TODO: find a good name for this
        m = np.zeros_like(latgr, dtype=np.float32)

        return latgr, longr, m

    @staticmethod
    def _heatmap_intensity(m):
        mmin = np.min(m)
        mmax = np.max(m)
        return 1.0 - (m - mmin) / (mmax - mmin)

    def _compute_rec_lags(self):
        if self._rx_lags:
            return

        for a, b in itertools.combinations(range(len(self._recs)), 2):
            lag_time, sigma2 = self._compute_recording_lags(self._recs[a], self._recs[b])
            self._rx_lags[(a, b)] = self._get_dist_probability_fn(lag_time, sigma2)
    
    def _get_heatmap(self, recids: list[tuple[int, int]]):
        m = np.copy(self._m)
        for (a, b) in recids:
            sigma2 = self._rx_lags[(a, b)]
            d1 = tools.haversine(self._latgr, self._longr, self._recs[a].lat, self._recs[a].lon)
            d2 = tools.haversine(self._latgr, self._longr, self._recs[b].lat, self._recs[b].lon)

            dist = d1 - d2

            m += sigma2(dist)

        return self._heatmap_intensity(m)

    def run(self):
        self._compute_rec_lags()

        intensity = self._get_heatmap(self._rx_lags.keys())

        return self._latgr, self._longr, intensity
