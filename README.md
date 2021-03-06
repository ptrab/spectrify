# spectrify.py
Parser for Excited State calculations using Gaussian 09/16, ORCA, and ADF:

| Progam  | spin-free | spin-orbit |
|---------|-----------|------------|
| G09/G16 | `--gaussian-out`* |   |
| ORCA    | `--orca-out` | `--orca-soc` |
| ADF     | `--adf-out` | `--adf-soc` |

*\* for Gaussian files there exists a mutually exclusive `--gaussian-singlet-triplet` command, which reaplces all oscillator strengths with an arbitrary value of 0.1 to make them visible*

Spectrify is written to generate broadened spectra from calculated excitation energies and oscillator strengths like
the Gaussian whitepaper http://www.gaussian.com/g_whitepap/tn_uvvisplot.htm and the cited paper (DOI: 10.1002/chir.20733) suggests.

The current output is only a matplotlib/pyplot spectrum that is saved if wished.

## Usage
Have a look into the help menu to find the options:
```bash
  ./spectrify -h
```
