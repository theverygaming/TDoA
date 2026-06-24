import argparse
import sys
import pathlib
import re
import itertools
import json
import io
import traceback
import base64
import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

import kiwiwavreader
import tools
import tdoa
import hfsim
import ionomodel


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

_KIWIRECORDER_REC_REGEX = r"\d{4}\d{2}\d{2}T\d{2}\d{2}\d{2}Z_\d+_([^_]+)_iq"

def _kiwirecorder_fname_name(name: str):
    return re.sub(r"-", "_", name)

def _recs_from_kiwirecorder_files(files: list[str], locs: dict[str, tuple[float, float]], locs_nameonly = False):
    recordings = []
    for fname in files:
        with open(fname, "rb") as f:
            m = re.search(_KIWIRECORDER_REC_REGEX, fname)
            rx_name = m.group(1) if m is not None else None
            if locs_nameonly and rx_name is None:
                raise Exception(f"could not match filename of recording {fname} and locs_nameonly is True, so the name is required")
            try:
                rx_coords = locs[fname if not locs_nameonly else rx_name]
                recordings.append(tdoa.TDoAPositionedRecording.from_recording(kiwiwavreader.read_kiwiwav(f), *rx_coords, name=rx_name))
            except Exception:
                print(f"error reading recording {fname}, ignoring")
                print(traceback.format_exc())
    return recordings

def _recs_from_directtdoa_dir(dir: str, directtdoa_db: str):
    with open(directtdoa_db, "r", encoding="utf-8") as f:
        directtdoa_db = json.loads(f.read())
    wavs = [str(fp) for fp in pathlib.Path(dir).glob("*.wav")]
    return _recs_from_kiwirecorder_files(
        wavs,
        {_kiwirecorder_fname_name(entry["id"]): (float(entry["lat"]), float(entry["lon"])) for entry in directtdoa_db},
        locs_nameonly=True,
    )

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
        m = re.match(_KIWIRECORDER_REC_REGEX, fname)
        fp = str(pathlib.Path(dir) / fname)
        name = _kiwirecorder_fname_name(m.group(1))
        locs[fp] = rxmap[name]
        files.append(fp)

    return _recs_from_kiwirecorder_files(files, locs)

def calc_nrows(ncols, nitems):
    nrows = math.ceil(nitems / ncols)
    return nrows

def _run_tdoa_heatmap_plot(latgr, longr, intensities, ncols, use_cartopy):
    lat_min, lat_max = np.min(latgr), np.max(latgr)
    lon_min, lon_max = np.min(longr), np.max(longr)
    heatmap_aspect_ratio = (abs(lon_max - lon_min) / abs(lat_max - lat_min)) if abs(lat_max - lat_min) > 0 else 1
    base_hmplot_dim = 10
    if heatmap_aspect_ratio >= 1:
        # wide
        hmplot_width = base_hmplot_dim * heatmap_aspect_ratio
        hmplot_height = base_hmplot_dim
    else:
        # tall
        hmplot_width = base_hmplot_dim
        hmplot_height = base_hmplot_dim / heatmap_aspect_ratio

    if len(intensities) == 1:
        ncols = 1
    nrows = calc_nrows(ncols, len(intensities))
    fig, axs = plt.subplots(ncols=ncols,
        nrows=nrows,
        squeeze=False,
        layout="tight",
        figsize=(
            (hmplot_width * ncols) + (1.3 * ncols),
            hmplot_height * nrows
        ),
        subplot_kw={"projection": ccrs.PlateCarree()} if HAS_CARTOPY and use_cartopy else None,
    )

    if len(intensities) % 2 == 1:
        axs.flat[-1].set_axis_off()

    for i, (intensity, title, markers, mark_max) in enumerate(intensities):
        ax = axs.flat[i]
        if HAS_CARTOPY and use_cartopy:
            img = ax.contourf(longr, latgr, intensity, levels=50, cmap="viridis", transform=ccrs.PlateCarree())

            ax.set_aspect("equal", adjustable="box")
            gl = ax.gridlines(draw_labels=True, dms=False, x_inline=False, y_inline=False)
            gl.top_labels = False
            gl.right_labels = False

            ax.add_feature(cfeature.BORDERS, linestyle="-", edgecolor="white", linewidth=1)
            ax.add_feature(cfeature.COASTLINE, edgecolor="white", linewidth=1)
        else:
            ax.set_aspect("equal", adjustable="box")
            img = ax.contourf(longr, latgr, intensity, levels=50, cmap="viridis")
        ax.set_xlim(np.min(longr), np.max(longr))
        ax.set_ylim(np.min(latgr), np.max(latgr))
        ax.set_title(title)
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Longitude")
        fig.colorbar(
            img,
            cax=ax.inset_axes([1.04, 0.0, 0.05, 1.0]),
            label="Probability",
        )

        def mark(lat, lon, color, label, label_on_map):
            ax.scatter(lon, lat, c=color, label=label)
            if label_on_map:
                ax.annotate(label_on_map, (lon, lat), ha="center", c="white", clip_on=True)

        if mark_max:
            max_idx = np.unravel_index(np.argmax(intensity), intensity.shape)
            print(f"{title} max: {latgr[max_idx]}, {longr[max_idx]}")
            print(f"{title} there are {len(np.where(np.ravel(intensity) == intensity[np.unravel_index(np.argmax(intensity), intensity.shape)])[0])} max values")
            mark(latgr[max_idx], longr[max_idx], "red", "max", f"{latgr[max_idx]:.4f}, {longr[max_idx]:.4f}")

        for mname, ((mlat, mlon), mcolor) in markers.items():
            mark(mlat, mlon, mcolor, mname, mname if not mname.startswith("_") else mname[1:])

        ax.legend()

# TODO: maybe support OSM output? https://wiki.openstreetmap.org/wiki/Heat_maps

def run_tdoa(
    recordings: list[tdoa.TDoAPositionedRecording],
    p1: tuple[float, float],
    p2: tuple[float, float],
    demod: str | None = None,
    split_secs: int | None = None,
    propmodel: ionomodel.PropModel | None = None,
    markers: dict[str, tuple[tuple[float, float], str]] | None = None
):
    if markers is None:
        markers = {}

    ncols = 2
    with matplotlib.backends.backend_pdf.PdfPages("out/TDoA spectrograms.pdf") as pdf:
        for rec_chunk in tools.iter_chunks(recordings, 4):
            nrows = calc_nrows(ncols, len(rec_chunk))
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows)
            if len(rec_chunk) % 2 == 1:
                axs.flat[-1].set_axis_off()
            fig.set_figwidth(16)
            fig.set_figheight(5 * nrows)
            for i, r in enumerate(rec_chunk):
                axs.flat[i].set_title(f"Spectrogram {r.name}")
                r.plot_spectrogram(fig, axs.flat[i])
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    tdoa_algo = tdoa.TDoAAlgorithmSimple()

    intensity_split = None
    intensity_split_corrected = None
    if split_secs is not None:
        print("running split TDoA")
        tdoa.TDoARecording.sync_recs(recordings)
        recordings_split = [list(x) for x in zip(*[r.split(split_secs) for r in recordings])]
        for i, recs in enumerate(recordings_split):
            print(f"split running {i+1}/{len(recordings_split)}")
            tdoa_run = tdoa.TDoARun(tdoa_algo, recs, None, p1, p2, demod=demod)
            if intensity_split is not None:
                intensity_split += tdoa_run.get_all(None)
                if propmodel is not None:
                    intensity_split_corrected += tdoa_run.get_all(propmodel)
            else:
                intensity_split = tdoa_run.get_all(None)
                if propmodel is not None:
                    intensity_split_corrected = tdoa_run.get_all(propmodel)
        intensity_split /= len(recordings_split)

    print("running TDoA")
    tdoa_run = tdoa.TDoARun(tdoa_algo, recordings, None, p1, p2, demod=demod)
    print(f"max TDoA resolution: {tdoa_run.get_max_res()}m")
    latgr, longr = tdoa_run.get_grid()
    intensity = tdoa_run.get_all(None)
    intensities = tdoa_run.get_pairs()
    if propmodel is not None:
        intensity_corrected = tdoa_run.get_all(propmodel)
    else:
        intensity_corrected = None

    def gen_rec_markers(recs):
        return {"_" + rec.name if rec.name is not None else "unknown": ((rec.lat, rec.lon), "blue") for rec in recs}

    heatmaps = []

    heatmaps.append((intensity, "TDoA", markers | gen_rec_markers(recordings), True))
    if propmodel is not None:
        heatmaps.append((intensity_corrected, "TDoA (corrected)", markers | gen_rec_markers(recordings), True))
    if intensity_split is not None:
        heatmaps.append((intensity_split, "TDoA (split)", markers | gen_rec_markers(recordings), True))
        if propmodel is not None:
            heatmaps.append((intensity_split_corrected, "TDoA (split, corrected)", markers | gen_rec_markers(recordings), True))

    _run_tdoa_heatmap_plot(latgr, longr, heatmaps, 1, True)
    plt.savefig(f"out/TDoA heatmaps.pdf")
    plt.close()

    heatmaps = []
    with matplotlib.backends.backend_pdf.PdfPages("out/TDoA correlations.pdf") as pdf:
        for intensities_chunk in tools.iter_chunks(intensities.items(), 4):
            ncols = 2
            nrows = calc_nrows(ncols, len(intensities_chunk))
            fig, axs = plt.subplots(ncols=ncols, nrows=nrows)
            if len(intensities_chunk) % 2 == 1:
                axs.flat[-1].set_axis_off()
            fig.set_figwidth(12)
            fig.set_figheight(4 * nrows)
            for i, ((r1id, r2id), intensity) in enumerate(intensities_chunk):
                rec1 = tdoa_run.get_rec(r1id)
                rec2 = tdoa_run.get_rec(r2id)
                heatmaps.append((intensity, f"TDoA {rec1.name} - {rec2.name}", markers | gen_rec_markers([rec1, rec2]), False))

                axs.flat[i].set_title(f"Correlation {rec1.name} - {rec2.name}")
                tdoa_run.plot_correlation(fig, axs.flat[i], r1id, r2id)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    with matplotlib.backends.backend_pdf.PdfPages("out/TDoA correlation heatmaps.pdf") as pdf:
        for heatmaps_chunk in tools.iter_chunks(heatmaps, 4):
            _run_tdoa_heatmap_plot(latgr, longr, heatmaps_chunk, 2, True)
            pdf.savefig()
            plt.close()

    return (
        latgr,
        longr,
        intensity,
        intensity_corrected,
        intensity_split,
        intensity_split_corrected,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LynxTDoA",
    )
    parser.add_argument("--p1", help="top left map boundary (format: lat,lon e.g. 42.1, -123.4)", type=str, required=True)
    parser.add_argument("--p2", help="bottom right map boundary (format: lat,lon e.g. 32.1, -87.65)", type=str, required=True)
    parser.add_argument("--input", "-i", help="input kiwirecorder directory", type=str, required=True)
    parser.add_argument("--demod", help="demodulate signal before correlation (default: disabled (complex correlation))", choices=["phase", "fm", "am"])
    parser.add_argument("--split", help="split run into segments of N seconds", type=int)
    parser.add_argument("--iono-height", help="virtual height of ionosphere in kilometers", type=float)
    parser.add_argument("--iono-takeoff", help="minimum takeoff angle for prop model in degrees (default: 0)", type=float, default=0.0)
    parsed_args = parser.parse_args()

    p1 = (float((split := parsed_args.p1.split(","))[0]), float(split[1]))
    p2 = (float((split := parsed_args.p2.split(","))[0]), float(split[1]))
    demod = parsed_args.demod
    split_secs = parsed_args.split
    propmodel_vh = parsed_args.iono_height
    propmodel_min_angle = parsed_args.iono_takeoff
    kiwirecorder_dir = parsed_args.input

    recordings = _recs_from_kiwirecorder_dir(kiwirecorder_dir)
    recordings = prepare_recs(recordings)
    run_tdoa(
        recordings,
        p1,
        p2,
        demod=demod,
        split_secs=split_secs,
        propmodel=ionomodel.SuperSimplePropModel(propmodel_vh, propmodel_min_angle) if propmodel_vh is not None else None,
        markers=None,
    )
