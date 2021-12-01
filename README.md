# spectrify.py
Parser for Excited State calculations using Gaussian 09/16, ORCA, and ADF:

| Progam  | spin-free | spin-orbit |
|---------|-----------|------------|
| G09/G16 | `--gaussian-out`* |   |
| ORCA 4  | `--orca-out` | `--orca-soc` |
| ORCA 5  | `--orca-out` | `--orca5-soc` |
| ADF     | `--adf-out` | `--adf-soc` |

*\* for Gaussian files there exists a mutually exclusive `--gaussian-singlet-triplet` command, which reaplces all oscillator strengths with an arbitrary value of 0.1 to make them visible*

Spectrify is written to generate broadened spectra from calculated excitation energies and oscillator strengths like
the Gaussian whitepaper http://www.gaussian.com/g_whitepap/tn_uvvisplot.htm and the cited paper (DOI: 10.1002/chir.20733) suggests.

The current output is only a matplotlib/pyplot spectrum that is saved if wished.

When I updated to python 3.9, the adjustText library failed to place the labels correctly.
My workaround is to setup a conda environment with python 3.7.
There it works as expected.

## Usage
Have a look into the help menu to find the options:
```bash
  ./spectrify -h
```
