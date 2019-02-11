# spectrify.py
Parser for Excited State calculations using Gaussian 09/16 (`--gaussian-out`) and ORCA (xy files which contain the spectrum table from the output `--orca-xy`, or simply the output file itself `--orca-out` as well as ORCA SOC spectra `--orca-soc`).

## usage
Have a look into the help menu to find the options:
```bash
  ./spectrify -h
```
It is written to generate broadened spectra from calculated excitation energies and oscillator strengths like
the Gaussian whitepaper http://www.gaussian.com/g_whitepap/tn_uvvisplot.htm and the cited paper (DOI: 10.1002/chir.20733) suggests.

The current output is only a matplotlib/pyplot spectrum that is saved if wished.
