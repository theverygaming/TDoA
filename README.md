# TDoA

_(experimental) TDoA program_

This is a small Python program to compute [TDoA](https://en.wikipedia.org/wiki/Time_of_arrival) with data from KiwiSDR receivers. It's very likely to still change a lot so definitely don't rely on this being very stable or usable.

## How do I use this?

Before running this you may want to install python 3, numpy, scipy and matplotlib. Optionally you may install [cartopy](https://cartopy.readthedocs.io/stable/index.html) for nicer maps.

At the time of writing there's a script `main.py` that will simply spit out it's usage when ran as is. It takes a few parameters, a path to a [kiwirecorder](https://github.com/jks-prv/kiwiclient) directory and will output plots into a directory named `out` (which you should create before running it).

Example kiwirecorder usage:
```sh
$ python3 kiwirecorder.py --kiwi-wav -d your_output_directory -s kiwisdrA.ddns.net,kiwisdrB.com,kiwisdrC.net -p 8073,8075,8073 --station kiwiA,kiwiB,kiwiC -L-5000 -H 5000 -f 4321 -m iq
```
- `-s`, `-p` and `--station` specify the kiwi hostnames, ports and names (the names will show up in the TDoA later)
- `-d xyz` specifies the output directory, you'll want to pass this to the TDoA script later.
- `-f 4321` specifies the frequency, in kHz
- `-L-5000 -H 5000` specifies the passband in Hz, in this case 5kHz below and above the carrier frequency. The passband should ideally match the bandwidth of the signal of interest closely.

for further help consult the [KiwiClient documentation on timestamped output](https://github.com/jks-prv/kiwiclient#iq-wav-files-with-gnss-timestamps) and `kiwirecorder.py --help`
