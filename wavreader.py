import struct
import numpy as np

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
    print(sr)

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

    sr_corrected = float(sr)
    CORR_FACTOR = 0.1
    gpsns_prev = 0
    for chunk in chunks_raw:
        if gpsns_prev > 0:
            prev_chunk_sr = len(chunk["data"]) / ((chunk["gpsns"] - gpsns_prev) * 1e-9)
            sr_corrected = ((1 - CORR_FACTOR) * sr_corrected) + (CORR_FACTOR * prev_chunk_sr)
        gpsns_prev = chunk["gpsns"]

    print(f"SR: {sr} SR corrected: {sr_corrected}")

# FIXME:
import sys

with open(sys.argv[1], "rb") as f:
    read_kiwiwav(f)
