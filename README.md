# spectrify.py
Parser for Excited State calculations using Gaussian 09 (Rev. A.02) and for the convolution of simple nm-f-lists.
The output is compatible with the LaTeX-Package pgfplots.

## usage
Have a look into the help menu to find the options:
```bash
  ./spectrify -h
```
It is written to generate broadened spectra from calculated excitation energies and oscillator strengths like
the Gaussian whitepaper http://www.gaussian.com/g_whitepap/tn_uvvisplot.htm and the cited paper (DOI: 10.1002/chir.20733) suggests.

The output will have five columns, `icm eV  nm  eps f`, which represent inverse cm (cm**-1), electron volt, nanometer,
molar attenuation coefficient ε and oscillator strength “unit”. The standard output is to `spectrum.dat`.
