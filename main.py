import sys
import pathlib
import re
import itertools
import json
import io
import traceback
import base64
import numpy as np
import scipy
import matplotlib.pyplot as plt
import kiwiwavreader
import tools
import tdoa
import hfsim

def prepare_recs(recs: list[tdoa.TDoAPositionedRecording]):
    recs_stitch = {}
    for rec in recs:
        if rec.name not in recs_stitch:
            recs_stitch[rec.name] = []
        recs_stitch[rec.name].append(rec)

    recs = []
    for k, v in recs_stitch.items():
        if len(v) == 1:
            recs.append(v[0])
            continue
        print(f"stitching {len(v)} recordings for '{k}'")
        try:
            recs.append(tdoa.TDoAPositionedRecording.stitch(v))
        except Exception:
            print(f"error stitching recodings from '{k}', ignoring")
            print(traceback.format_exc())

    tdoa.TDoARecording.sync_recs(recs)

    return recs


def _recs_from_files(files: list[str], locs: dict[str, tuple[float, float]]):
    recordings = []
    for fname in files:
        with open(fname, "rb") as f:
            m = re.search(r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}Z_\d+_([^_]+)_iq", fname)
            try:
                recordings.append(tdoa.TDoAPositionedRecording.from_recording(kiwiwavreader.read_kiwiwav(f), *locs[fname], name=m.group(1) if m is not None else None))
            except Exception:
                print(f"error reading recording {fname}, ignoring")
                print(traceback.format_exc())
    return recordings

def _recs_from_kiwirecorder_dir(dir: str, wavs: list[str] | None = None):
    if wavs is None:
        wavs = [fp.name for fp in pathlib.Path(dir).glob("*.wav")]
        print("processing kiwirecorder wavs: ", json.dumps(wavs, indent=4))

    rxmap = {}
    for fp in pathlib.Path(dir).glob("*.txt"):
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read()
        m = re.match(r"d\.(?P<name>[^ ]+) = struct\(.*'coord', \[(?P<lat>-?\d+\.\d+),(?P<lon>-?\d+\.\d+)\]", content)
        rxmap[m.group("name")] = (float(m.group("lat")), float(m.group("lon")))

    locs = {}
    files = []
    for fname in wavs:
        m = re.match(r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}Z_\d+_([^_]+)_iq", fname)
        fp = str(pathlib.Path(dir) / fname)
        locs[fp] = rxmap[m.group(1)]
        files.append(fp)

    return _recs_from_files(files, locs)

def _plot_tdoa_heatmap(out, latgr, longr, intensity, title, markers, mark_max=True):
    # TODO: maybe support OSM output? https://wiki.openstreetmap.org/wiki/Heat_maps

    plt.figure(figsize=(16, 12))
    plt.contourf(longr, latgr, intensity, levels=50, cmap="viridis")
    plt.xlim(np.min(longr), np.max(longr))
    plt.ylim(np.min(latgr), np.max(latgr))
    plt.title(title)
    plt.xlabel("Latitude")
    plt.ylabel("Longitude")
    plt.colorbar(label="Probability")

    def mark(lat, lon, color, label, label_on_map):
        plt.scatter(lon, lat, c=color, label=label)
        if label_on_map:
            plt.annotate(label_on_map, (lon, lat), ha="center", c="white")

    if mark_max:
        max_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
        print(f"{title} max: {latgr[max_idx]}, {longr[max_idx]}")
        print(f"{title} there are {len(np.where(np.ravel(intensity) == intensity[np.unravel_index(np.argmax(intensity), intensity.shape)])[0])} max values")
        mark(latgr[max_idx], longr[max_idx], "red", "max", f"{latgr[max_idx]:.4f}, {longr[max_idx]:.4f}")

    for mname, ((mlat, mlon), mcolor) in markers.items():
        mark(mlat, mlon, mcolor, mname, mname)

    plt.legend()

    plt.savefig(f"{out}{title}.png")
    plt.close()


def run_tdoa(
    recordings: list[tdoa.TDoAPositionedRecording],
    p1: tuple[float, float],
    p2: tuple[float, float],
    demod: str | None = None,
    split_secs: int | None = None,
    markers: dict[str, tuple[tuple[float, float], str]] | None = None
):
    if markers is None:
        markers = {}

    for r in recordings:
        fig, ax = plt.subplots()
        ax.set_title(f"Spectrogram {r.name}")
        r.plot_spectrogram(fig, ax)
        plt.tight_layout()
        plt.savefig(f"out/{r.name} spec.png")
        plt.close()

    tdoa_algo = tdoa.TDoAAlgorithmSimple(demod=demod)

    intensity_split = None
    if split_secs is not None:
        print("running split TDoA")
        tdoa.TDoARecording.sync_recs(recordings)
        recordings_split = [list(x) for x in zip(*[r.split(split_secs) for r in recordings])]
        for i, recs in enumerate(recordings_split):
            print(f"split running {i+1}/{len(recordings_split)}")
            tdoa_run = tdoa.TDoARun(tdoa_algo, recs, None, p1, p2)
            if intensity_split is not None:
                intensity_split += tdoa_run.get_all()
            else:
                intensity_split = tdoa_run.get_all()
        intensity_split /= len(recordings_split)

    print("running TDoA")
    tdoa_run = tdoa.TDoARun(tdoa_algo, recordings, None, p1, p2)
    print(f"max TDoA resolution: {tdoa_run.get_max_res()}m")
    latgr, longr = tdoa_run.get_grid()
    intensity = tdoa_run.get_all()
    intensities = tdoa_run.get_pairs()

    def gen_rec_markers(recs):
        return {rec.name if rec.name is not None else "unknown": ((rec.lat, rec.lon), "blue") for rec in recs}

    _plot_tdoa_heatmap("out/", latgr, longr, intensity, "TDoA", markers | gen_rec_markers(recordings))
    if intensity_split is not None:
        _plot_tdoa_heatmap("out/", latgr, longr, intensity_split, "TDoA (split)", markers | gen_rec_markers(recordings))

    for (r1id, r2id), intensity in intensities.items():
        rec1 = tdoa_run.get_rec(r1id)
        rec2 = tdoa_run.get_rec(r2id)
        _plot_tdoa_heatmap("out/", latgr, longr, intensity, f"TDoA {rec1.name} - {rec2.name}", markers | gen_rec_markers([rec1, rec2]), mark_max=False)

        fig, ax = plt.subplots()
        ax.set_title(f"Correlation {rec1.name} - {rec2.name}")
        tdoa_run.plot_correlation(fig, ax, r1id, r2id)
        plt.tight_layout()
        plt.savefig(f"out/TDoA {rec1.name} - {rec2.name} correlation.png")
        plt.close()

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print(f"usage: {sys.argv[0]} <lat top left> <lon top left> <lat bottom right> <lon bottom right> <demod type or 'none'> <split seconds or 0 for no split> <kiwirecorder dir>")
        exit(1)
    p1 = (float(sys.argv[1]), float(sys.argv[2]))
    p2 = (float(sys.argv[3]), float(sys.argv[4]))
    demod = sys.argv[5] if sys.argv[5].lower() != "none" else None
    split_secs = int(sys.argv[6]) if int(sys.argv[6]) != 0 else None
    kiwirecorder_dir = sys.argv[7]

    recordings = _recs_from_kiwirecorder_dir(kiwirecorder_dir)
    recordings = prepare_recs(recordings)
    run_tdoa(
        recordings,
        p1,
        p2,
        demod=demod,
        split_secs=split_secs,
        markers=None,
    )
