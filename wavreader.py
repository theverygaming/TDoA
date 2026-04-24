import struct
import numpy as np
from tdoa import TDoARecording


def _read_chunk(data):
    ctype = data[0:4]
    match ctype:
        case b"kiwi":
            lchunk, gps_last, dummy, gpssec, gpsnsec = struct.unpack("<IBBII", data[4:18])
            if lchunk != 10:
                raise Exception("invalid kiwi chunk length")
            cdata = {
                "chunk_type": "kiwi",
                "fix_age": gps_last,
                "gpssec": gpssec,
                "gpsnsec": gpsnsec,
            }
        case b"data":
            lchunk, = struct.unpack("<I", data[4:8])
            data = data[8:8+lchunk]
            if len(data) != lchunk:
                raise Exception("data chunk too short")
            cdata = {
                "chunk_type": "data",
                "data": data,
            }
        case _:
            raise Exception(f"unknown chunk type {ctype}")
    return (4 + 4 + lchunk, cdata)

def read_kiwiwav(f):
    data = f.read()
    if data[0:4] != b"RIFF":
        raise Exception("not a RIFF")
    fsize, = struct.unpack("<I", data[4:8])
    if data[8:12] != b"WAVE":
        raise Exception("not a WAVE")
    if data[12:16] != b"fmt ":
        raise Exception("no fmt chunk found")
    lfmt, fmtt, nchannels, sr, br, ba, bits = struct.unpack("<IHHIIHH", data[16:36])
    if lfmt != 16:
        raise Exception("invalid fmt chunk length")
    if nchannels != 2:
        raise Exception("expected 2 channels")
    if bits != 16:
        raise Exception("expected 16-bit samples")

    chunks_raw = []
    ridx = 36
    while ridx < len(data):
        cl, cdata1 = _read_chunk(data[ridx:])
        ridx += cl
        if cdata1["chunk_type"] != "kiwi":
            raise Exception("expected kiwi chunk")
        cl, cdata2 = _read_chunk(data[ridx:])
        ridx += cl
        if cdata2["chunk_type"] != "data":
            raise Exception("expected data chunk")
        chunks_raw.append({
            "fix_age": cdata1["fix_age"],
            "gpsns": (cdata1["gpssec"] * int(1e9)) + cdata1["gpsnsec"],
            "data": (np.frombuffer(cdata2["data"], dtype="<i2").astype(np.float32) / 32768).view(np.complex64),
        })

    chunk_len = len(chunks_raw[0]["data"])
    if len(set(len(x["data"]) for x in chunks_raw)) != 1:
        raise Exception("unsupported: chunks have different lengths")

    dt_ns_chunks = [] 
    gpsns_prev = 0
    for i, chunk in enumerate(chunks_raw):
        if i == 0 and chunk["gpsns"] != 0:
            raise Exception("first chunk has nonzero timestamp")
        if chunk["gpsns"] == 0 and i != 0:
            raise Exception("zero timestamp after start")
        if gpsns_prev > 0:
            if chunk["gpsns"] < gpsns_prev:
                raise Exception("clock went backwards or didn't run")
            dt_ns = chunk["gpsns"] - gpsns_prev
            dt_ns_chunks.append(dt_ns)
        gpsns_prev = chunk["gpsns"]
    if len(dt_ns_chunks) == 0:
        raise Exception("no usable timestamps")
    if len(dt_ns_chunks) + 2 != len(chunks_raw):
        raise Exception("dt_ns_chunks length doesn't fit together with chunks_raw length")
    dt_ns_median = np.median(dt_ns_chunks)
    sr_measured = chunk_len / (dt_ns_median * 1e-9)
    sr_ppm = ((sr_measured - sr) / sr) * 1e6

    # guessed timestamp for first chunk
    chunks_raw[0]["gpsns"] = chunks_raw[1]["gpsns"] - dt_ns_median

    # add length of chunk to each chunk where possible, estimate for others (first & last)
    for i, chunk in enumerate(chunks_raw):
        if i < 1:
            chunk["duration_ns"] = dt_ns_median
        elif i == len(chunks_raw) - 1:
            chunk["duration_ns"] = dt_ns_median
        else:
            chunk["duration_ns"] = dt_ns_chunks[i - 1]

    # timing sanity check
    for i, chunk in enumerate(chunks_raw):
        if i == len(chunks_raw) - 1:
            break
        dt = chunks_raw[i+1]["gpsns"] - (chunk["gpsns"] + chunk["duration_ns"])
        if dt != 0:
            raise Exception(f"chunks {i} and {i+1} are spaced weirdly")

    # compute timestamps for each sample
    for chunk in chunks_raw:
        samp_dt_ns = chunk["duration_ns"] / len(chunk["data"])
        chunk["timestamps"] = chunk["gpsns"] + (np.arange(chunk_len, dtype=np.int64) * samp_dt_ns)

    samples_all = np.concatenate([x["data"] for x in chunks_raw])
    timestamps_all = np.concatenate([x["timestamps"] for x in chunks_raw])

    # last sanity check
    if len(samples_all) != len(timestamps_all):
        raise Exception("len(samples_all) != len(timestamps_all)")

    return TDoARecording(
        samples_all,
        timestamps_all,
        sr_measured,
    )
