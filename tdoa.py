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
    def sync_recs(recs: list["TDoARecording"], time_diff_max_ns=2e9):
        """
        Throws away all parts of the specified recordings that are not overlapping.

        Operates on references of the recordings!
        """

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

    def get_dist_score_fn(self, r1: TDoARecording, r2: TDoARecording):
        raise NotImplementedError()

    def score_to_intensity(self, score: npt.NDArray[np.float32]):
        raise NotImplementedError()


class TDoAAlgorithmSimple(TDoAAlgorithm):
    def __init__(self, max_dist_m=10000*1000):
        self._max_dist_m = max_dist_m

    def get_dist_score_fn(self, r1: TDoARecording, r2: TDoARecording):
        lag_time, score = self._compute_recording_lags(r1, r2)
        return self._get_dist_score_fn(lag_time, score)

    def score_to_intensity(self, score: npt.NDArray[np.float32]):
        scoremin = np.min(score)
        scoremax = np.max(score)
        return (score - scoremin) / (scoremax - scoremin)

    @staticmethod
    def _compute_lags(s1, s2, sr, max_dist_m):
        # this is in essence similar to
        # https://github.com/hcab14/TDoA/blob/2bb9dc2ecc2c6ebcc13ed11c7cbeadea0cd5dfcd/m/tdoa_compute_lags_new.m#L20-L28
        # and ofc strongly inspired by that

        # get phase
        #s1 = np.angle(s1) / (2 * np.pi)
        #s2 = np.angle(s2) / (2 * np.pi)
        # get magnitude (AM demod)
        # s1 = np.abs(s1)
        # s2 = np.abs(s2)

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

        score = np.log(np.abs(corr))

        return lag_time, score

    def _compute_recording_lags(self, r1, r2):
        start_offset = (r1.timestamps[0] - r2.timestamps[0]) / 1e9
        lag_time, score = self._compute_lags(
            r1.samples,
            r2.samples,
            np.mean([r1.sr, r2.sr]),
            self._max_dist_m,
        )
        lag_time += start_offset
        return lag_time, score

    @staticmethod
    def _get_dist_score_fn(lag_time, score):
        # convert seconds lag to distance in m
        lag_time *= scipy.constants.c
        # function that will, given a distance in meters return the score at that point
        return scipy.interpolate.interp1d(
            lag_time,
            score,
            bounds_error=False,
            fill_value=np.min(score),
        )


class TDoARun:
    def __init__(self, algorithm: TDoAAlgorithm, recs: list[TDoAPositionedRecording], p1, p2, res):
        self._algorithm = algorithm
        self._recs = recs
        self._rx_dist_fns = {}
        self._latgr, self._longr, self._score_template = self._prepare_heatmap(p1, p2, res)

        if len(self._recs) < 2:
            raise Exception(f"need at least two recordings for TDoA, got {len(self._recs)}")

        TDoARecording.sync_recs(self._recs)

        for r in self._recs:
            r.resample(100, 1)

    @staticmethod
    def _prepare_heatmap(p1, p2, res):
        lat_min = min(p1[0], p2[0])
        lat_max = max(p1[0], p2[0])
        lon_min = min(p1[1], p2[1])
        lon_max = max(p1[1], p2[1])

        lats = np.arange(lat_min, lat_max, res)
        lons = np.arange(lon_min, lon_max, res)

        longr, latgr = np.meshgrid(lons, lats)

        score = np.zeros_like(latgr, dtype=np.float32)

        return latgr, longr, score

    def _compute_rec_dists(self):
        if self._rx_dist_fns:
            return

        for a, b in itertools.combinations(range(len(self._recs)), 2):
            self._rx_dist_fns[(a, b)] = self._algorithm.get_dist_score_fn(self._recs[a], self._recs[b])

    def _get_heatmap(self, recpairs: list[tuple[int, int]]):
        score = np.copy(self._score_template)
        for (a, b) in recpairs:
            d1 = tools.haversine(self._latgr, self._longr, self._recs[a].lat, self._recs[a].lon)
            d2 = tools.haversine(self._latgr, self._longr, self._recs[b].lat, self._recs[b].lon)

            dist = d1 - d2

            score += self._rx_dist_fns[(a, b)](dist)

        return self._algorithm.score_to_intensity(score)

    def get_pairs(self):
        self._compute_rec_dists()
        intensities = {}
        for recids in self._rx_dist_fns:
            intensities[recids] = self._get_heatmap([recids])
        return intensities

    def get_all(self):
        self._compute_rec_dists()

        intensity = self._get_heatmap(self._rx_dist_fns.keys())

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
