#!/usr/bin/env python3
import sys
import argparse
import numpy as np

import adjustText as aT

import matplotlib as mpl
import matplotlib.patheffects as path_effects

import matplotlib.pyplot as plt

SMALL_SIZE = 11
MEDIUM_SIZE = 12
BIGGER_SIZE = 13

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

mpl.rcParams['font.family'] = ['Noto Sans Display', 'Noto Sans', 'sans-serif']

def getinput(args):
    """parse the input"""
    parser = argparse.ArgumentParser(
        description=("Gaussian/ORCA/ADF to UV-Vis-spectrum converter.")
    )
    parser.add_argument(
        "--gaussian-out",
        "-gout",
        nargs="*",
        help=("Typically *.log or *.out, but ending doesn't matter."),
    )
    parser.add_argument(
        "--gaussian-singlet-triplet",
        "-gst",
        action="store_true",
        help=(
            "sets oscillator strengths to arbitrary value of 0.1 to make them visible. Mutually exclusive with regular gaussian output!"
        ),
    )
    parser.add_argument(
        "--half-width-at-half-maximum",
        "-hwhm",
        "-w",
        default=0.333,
        type=float,
        help=("half width at half maximum in eV (default: 0.333)"),
    )
    parser.add_argument(
        "--nm-min",
        "-i",
        default=280.0,
        type=float,
        help=("minimal wavelength in nm (default: 280.0)"),
    )
    parser.add_argument(
        "--nm-max",
        "-f",
        default=780.0,
        type=float,
        help=("maximal wavelength in nm (default: 780.0"),
    )
    parser.add_argument(
        "--y-height",
        "-y",
        type=float,
        default=0.0,
        help="change the maximum value of the y axis"
    )
    parser.add_argument(
        "--step-size",
        "-s",
        "-dx",
        default=1.0,
        type=float,
        help=("stepsize of the spectrum in nm(default: 1.0)"),
    )
    parser.add_argument(
        "--label-nm-min",
        "-li",
        type=float,
        help=("minimal wavelength in nm for labeling (default: 1.1 * nm_min)"),
    )
    parser.add_argument(
        "--label-nm-max",
        "-lf",
        type=float,
        help=("maximal wavelength in nm for labeling (default: 0.9 * nm_max)"),
    )
    parser.add_argument(
        "--spectrum-out",
        "--out",
        "-o",
        default="spectrum.svg",
        help=("outputfile for the spectrum (default: spectrum.svg)"),
    )
    parser.add_argument(
        "--no-save", "-nos", action="store_true", help=("to store the plot or not")
    )
    parser.add_argument(
        "--no-plot", "-nop", action="store_true", help=("to plot the plot or not")
    )
    orca_group = parser.add_mutually_exclusive_group(required=False)
    orca_group.add_argument(
        "--orca-xy",
        "-oxy",
        nargs="*",
        help="takes a Excited States Table from ORCA without the header",
    )
    # orca_group.add_argument(
    #     "--print-s-squared",
    #     "-s2",
    #     action="store_true",
    #     help="prints a table for gaussian files with the spin contamination",
    # )
    parser.add_argument("--orca-out", "-oout", nargs="*", help="orca output files")
    parser.add_argument(
        "--orca-soc", "-osoc", nargs="*", help="orca output files, reading the SOC states"
    )
    parser.add_argument(
        "--orca5-soc", "-o5soc", nargs="*", help="orca 5 output files, reading the SOC states"
    )
    legend_group = parser.add_mutually_exclusive_group(required=False)
    legend_group.add_argument(
        "--plot-legend",
        "-leg",
        nargs="*",
        help="plot legend entries, orca out > orca xy > orca soc > orca 5 soc > gaussian out > adf out > adf soc > exp",
    )
    legend_group.add_argument(
        "--no-legend",
        "-nol",
        action="store_true",
        help="if no legend is wanted, default: false",
    )
    parser.add_argument(
        "--top-eV", "-tev", action="store_true", help="gives additional eV axis at top"
    )
    parser.add_argument(
        "--peak-labels", "-label", action="store_true", help="peak labels"
    )
    parser.add_argument(
        "--peak-prefixes",
        "-pp",
        nargs="*",
        default=[],
        help="add prefixes to the labels, orca out > orca xy > orca soc > orca 5 soc > gaussian out > adf out > adf soc",
    )
    parser.add_argument(
        "--fosc-threshold",
        "-ft",
        type=float,
        default=0.05,
        help="sets the oscillator strength threshold for labeling (default: 0.05)",
    )
    parser.add_argument("--adf-out", "-aout", nargs="*", help="adf output files")
    parser.add_argument("--adf-soc", "-asoc", nargs="*", help="adf soc state files")
    parser.add_argument(
        "--exp", "-exp", default=False, nargs="*", help="add experimental spectra to the plot."
    )
    parser.add_argument(
        "--exckel",
        nargs=3,
        metavar=("Conv_FOsc", "Filtered_Peaks", "Spectrum_Peaks"),
        help="interface for Exckel",
    )
    parser.add_argument("--exckel-grid", action="store_true", help=("grid for exckel"))
    parser.add_argument(
        "--exckel-color",
        default="k",
        help=("color for exckel spectrum in matplotlib compatible format (default: k)"),
    )
    parser.add_argument(
        "--dots-per-inch",
        "-dpi",
        default=100,
        type=int,
        help=("dpi for the exported spectrum (default: 100)"),
    )
    # parser.add_argument(
    #     "--export-spectra",
    #     "-es",
    #     action="store_true",
    #     help=("exports the plotted spectra to xy-files")
    # )
    parser.add_argument(
        "--colors",
        "-clr",
        default=None,
        nargs="*",
        help=("specify your own colors in matplotlib style https://matplotlib.org/stable/tutorials/colors/colors.html"),
    )
    parser.add_argument(
        "--linestyles",
        "-ls",
        nargs="*",
        default=[],
        help=("specify the linestyles of the lines"),
    )
    parser.add_argument(
        "--aspect-ratio",
        "-ratio",
        default=np.sqrt(2),
        type=float,
        help=("specify the aspect ratio of the plot (default sqrt(2))"),
    )
    parser.add_argument(
        "--tertiary-axis",
        "-tert",
        action="store_true",
        help="Adds a tertiary y-axis for experimental spectra's extinction coefficient. (default: False)"
    )
    parser.add_argument(
        "--y2-height",
        "-y2",
        type=float,
        default=0.0,
        help="Sets the height of the secondary y-axis (or of the tertiary y-axis if --tertiary-axis is set.)"
    )

    return parser.parse_args(args)


def main():
    args = getinput(sys.argv[1:])

    if args.no_save and args.no_plot:
        sys.exit("no plot and no save ... done \(^o ^)/")

    if args.exckel:
        plot_for_exckel(args)
        sys.exit()

    excited_states = []
    if args.orca_out:
        for file in args.orca_out:
            excited_states.append(get_orca_excited_states(file))
    if args.orca_xy:
        for file in args.orca_xy:
            excited_states.append(get_orca_xy_excited_states(file))
    if args.orca_soc:
        for file in args.orca_soc:
            excited_states.append(get_orca_soc_states(file))
    if args.orca5_soc:
        for file in args.orca5_soc:
            excited_states.append(get_orca5_soc_states(file))
    if args.gaussian_out:
        for file in args.gaussian_out:
            if args.gaussian_singlet_triplet:
                es = get_gaussian_excited_states(file)
                tes = es.T
                tes[3] = [0.1 for x in tes[3]]
                es = tes.T
                excited_states.append(es)
            else:
                excited_states.append(get_gaussian_excited_states(file))
    if args.adf_out:
        for file in args.adf_out:
            excited_states.append(get_adf_excited_states(file))
    if args.adf_soc:
        for file in args.adf_soc:
            excited_states.append(get_adf_soc_states(file))

    nm_grid = []
    oscillator_dist = []
    epsilon_dist = []
    for state_file in excited_states:
        tmp1, tmp2, tmp3 = broaden(state_file, args)
        nm_grid.append(tmp1)
        oscillator_dist.append(tmp2)
        epsilon_dist.append(tmp3)

    # if not args.gaussian_out:
    #     cnt = 0
    #     if args.print_s_squared:
    #         for state_file in excited_states:
    #             print(f"file {cnt}")
    #             for state in state_file:
    #                 add_s2 = spin_contamination_additive(1, state[-1])
    #                 mix_s2 = spin_contamination_mixed(1, state[-1])
    #                 add_s2 = np.around(100 * add_s2, decimals=2)
    #                 mix_s2 = np.around(100 * mix_s2, decimals=2)
    #                 print(f"{state} {add_s2} {mix_s2}")

    nm_grid = np.array(nm_grid)
    oscillator_dist = np.array(oscillator_dist)
    epsilon_dist = np.array(epsilon_dist)

    plot_spectra(nm_grid, oscillator_dist, epsilon_dist, excited_states, args)


def get_adf_excited_states(file):
    with open(file, "r") as handle:
        lines = handle.readlines()

    excited_state_list = []
    nexc = 0
    for line in lines:
        if "lowest" in line.lower():
            nexc = int(line.split()[-1])
            break

    cnt = 0
    for line in lines:
        if "All SINGLET-SINGLET excitation energies" in line:
            # All SINGLET-SINGLET excitation energies
            #
            #  no.     E/a.u.        E/eV      f           Symmetry
            #  -----------------------------------------------------
            #    1:     0.10771      2.93087   0.9699E-01  A
            #    0          1           2          3       4
            for state in lines[cnt + 4 : cnt + nexc + 4]:
                sline = state.replace(":", "").split()
                number = int(sline[0])
                energy = float(sline[2])
                wavelength = 1239.841_973_862_09 / energy
                oscillator_strength = float(sline[3])
                excited_state_list.append(
                    [number, energy, wavelength, oscillator_strength]
                )
            break
        cnt += 1

    return np.array(excited_state_list)


def get_adf_soc_states(file):
    with open(file, "r") as handle:
        lines = handle.readlines()

    excited_state_list = []
    nexc = 0
    gscorr = 0
    for line in lines:
        if "lowest" in line.lower():
            nexc = 4 * int(line.split()[-1])
        if "gscorr" in line.lower():
            gscorr = 1
        if "end input" in line.lower():
            break
    print(f"GSCORR: {gscorr}")

    cnt = 0
    for line in lines:
        if "All Spin-Orbital Coupling Excitation Energies" in line:
            # All Spin-Orbital Coupling Excitation Energies
            #
            #  no.     E/a.u.        E/eV      f           tau/s        Symmetry
            #  ------------------------------------------------------------------
            #    1:     0.00000      0.00000   0.3820E-07               A
            #    0         1             2          3                   4
            for state in lines[cnt + 4 + gscorr : cnt + nexc + 4 + gscorr]:
                sline = state.replace(":", "").split()
                number = int(sline[0])
                energy = float(sline[2])
                wavelength = 1239.841_973_862_09 / energy
                oscillator_strength = float(sline[3])
                excited_state_list.append(
                    [number, energy, wavelength, oscillator_strength]
                )
            break
        cnt += 1

    return np.array(excited_state_list)


def get_orca_xy_excited_states(file):
    with open(file, "r") as handle:
        lines = handle.readlines()

    excited_state_list = []
    for line in lines:
        # -----------------------------------------------------------------------------
        # State   Energy  Wavelength   fosc         T2         TX        TY        TZ
        #         (cm-1)    (nm)                  (au**2)     (au)      (au)      (au)
        # -----------------------------------------------------------------------------
        #   0        1         2         3          4           5         6         7
        sline = line.split()
        number = int(sline[0])
        energy = 10 ** 7 / float(sline[1])
        wavelength = float(sline[2])
        oscillator_strength = float(sline[3])
        excited_state_list.append([number, energy, wavelength, oscillator_strength])

    return np.array(excited_state_list)


def get_orca_excited_states(file):
    with open(file, "r") as handle:
        lines = handle.readlines()

        excited_state_list = []
        for i in range(len(lines)):
            if "ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS" in lines[i]:
                absorption_start = i + 5
            elif (
                "ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS" in lines[i]
            ):
                absorption_end = i - 2
                break

        for line in lines[absorption_start:absorption_end]:
            if "spin forbidden" in line:
                continue
            sline = line.split()
            number = int(sline[0])
            energy = 10 ** 7 / float(sline[1])
            wavelength = float(sline[2])
            oscillator_strength = float(sline[3])
            excited_state_list.append([number, energy, wavelength, oscillator_strength])

        return np.array(excited_state_list)


def get_orca_soc_states(file):
    with open(file, "r") as handle:
        lines = handle.readlines()

        excited_state_list = []
        for i in range(len(lines)):
            if (
                "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
                in lines[i]
            ):
                absorption_start = i + 5
            elif (
                "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"
                in lines[i]
            ):
                absorption_end = i - 2
                break

        for line in lines[absorption_start:absorption_end]:
            sline = line.split()
            number = int(sline[0])
            energy = 10 ** 7 / float(sline[1])
            wavelength = float(sline[2])
            oscillator_strength = float(sline[3])
            excited_state_list.append([number, energy, wavelength, oscillator_strength])

        return np.array(excited_state_list)

def get_orca5_soc_states(file):
    with open(file, "r") as handle:
        lines = handle.readlines()

        excited_state_list = []
        for i in range(len(lines)):
            if (
                "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS"
                in lines[i]
            ):
                absorption_start = i + 5
            elif (
                "SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION VELOCITY DIPOLE MOMENTS"
                in lines[i]
            ):
                absorption_end = i - 2
                break

        # ---------------------------------------------------------------------------------------------------------------------
        #                     SOC CORRECTED ABSORPTION SPECTRUM VIA TRANSITION ELECTRIC DIPOLE MOMENTS*
        # ---------------------------------------------------------------------------------------------------------------------
        #    State    Energy  Wavelength   fosc         T2              TX                    TY                    TZ
        #             (cm-1)    (nm)                  (au**2)          (au)                  (au)                  (au)
        # ---------------------------------------------------------------------------------------------------------------------
        #    0   1   15218.8    657.1   0.000001183   0.00003 ( -0.00154,  0.00033) (  0.00145, -0.00032) (  0.00447, -0.00097)
        #    0   1      2         3          4           5    6  7           8      9   10       11       12  13        14
        for line in lines[absorption_start:absorption_end]:
            sline = line.split()
            number = int(sline[1])
            energy = 10 ** 7 / float(sline[2])
            wavelength = float(sline[3])
            oscillator_strength = float(sline[4])
            excited_state_list.append([number, energy, wavelength, oscillator_strength])

        return np.array(excited_state_list)

def get_gaussian_excited_states(file):
    with open(file, "r") as handle:
        lines = handle.readlines()

    excited_state_list = []
    for line in lines:
        if "Excited State" in line:
            # Excited State   1:      Singlet-B1     8.3203 eV  149.01 nm  f=0.0247  <S**2>=0.000
            #   0      1      2         3              4    5     6     7     8         9
            sline = line.split()
            number = int(sline[2].replace(":", ""))
            energy = float(sline[4])
            wavelength = float(sline[6])
            oscillator_strength = float(sline[8].replace("f=", ""))
            spin_contamination = float(sline[9].replace("<S**2>=", ""))
            excited_state_list.append(
                [number, energy, wavelength, oscillator_strength, spin_contamination]
            )

    return np.array(excited_state_list)


def broaden(excited_states, args):
    eV2nm = 1239.841_973_862_09
    prefactor = 1.306_297_4

    standard_dev_eV = args.half_width_at_half_maximum / np.sqrt(2 * np.log(2))
    step_size = args.step_size

    nm_list = excited_states[:, 2]
    oscillator_strengths = excited_states[:, 3]

    nm_grid = np.arange(args.nm_min, args.nm_max + step_size / 2, step_size)
    oscillator_multipliers = oscillator_strengths
    # the 10 is the result of 10 ** 8 / 10 ** 7, from which the first was given in the
    # white paper as part of the prefactor and the latter comes from the conversion
    # between nm and cm^-1
    epsilon_multipliers = (
        10 * prefactor * eV2nm * oscillator_multipliers / standard_dev_eV
    )
    fractions = 1 / nm_list.reshape((-1, len(nm_list))).T - 1 / nm_grid
    exponentials = np.exp(-1.0 * (eV2nm / standard_dev_eV * fractions) ** 2)
    oscillator_dist = oscillator_multipliers.dot(exponentials)
    epsilon_dist = epsilon_multipliers.dot(exponentials)

    return nm_grid, oscillator_dist, epsilon_dist


def nearest_spin(spin):
    spin_list = np.arange(0, 20, 0.5)
    spin_list = [x * (x + 1) for x in spin_list]
    spin_list[spin_list < spin] = 0
    return np.max(spin_list)


def spin_contamination_additive(spin, s_squared):
    """ gives the contribution of the next higher contaminating state it assumes, that
    in spin contamination the next highest lying excited state has the greatest impact
    and that the mixing is additive, as with 100% CONTAMINATION one should have both
    states and not only the contaminating one

    <S²> = <S²> + k * <(S+1)²>

        <S²> - S * (S + 1)
    k = ------------------
         (S + 1) * (S + 2)
    """
    return (s_squared - spin * (spin + 1)) / ((spin + 1) * (spin + 2))


def spin_contamination_mixed(spin, s_squared):
    """ gives the contribution of the next higher contaminating state, but is based on
    a more mixing formula than the additive one above (based on a discussion with Kevin
    Fiederling). while the former equation was my interpretation, this seems to be in
    accordance with a Casida paper: 10.1016/j.theochem.2009.07.036

    it assumes, that in spin contamination the next highest
    lying excited state has the greatest impact but also
    that the mixing is not additive, as above

    <S²> = (1 - k) <S²> + k <(S+1)²>

        <S²> - S * (S + 1)
    k = ------------------
           2 * (S + 1)

    this value is always larger than the above formula
    by a factor of 1 + S / 2
    """
    return (s_squared - spin * (spin + 1)) / (2 * (spin + 1))


# from https://stackoverflow.com/a/10739207
def get_text_positions(x_data, y_data, text_widths, txt_height):
    a = list(zip(y_data, x_data))
    text_positions = y_data.copy()

    for index, (y, x) in enumerate(a):
        local_text_positions = [
            i
            for i in a
            if i[0] > (y - txt_height)
            and (abs(i[1] - x) < text_widths[index])  # * 2)
            and i != (y, x)
        ]

        if local_text_positions:
            sorted_ltp = sorted(local_text_positions)

            if abs(sorted_ltp[0][0] - y) < txt_height:  # True == collision
                differ = np.diff(sorted_ltp, axis=0)
                a[index] = (sorted_ltp[-1][0] + txt_height, a[index][1])
                text_positions[index] = sorted_ltp[-1][0] + txt_height

                for k, (j, m) in enumerate(differ):
                    # j is the vertical distance between words
                    if j > txt_height * 2:  # if True then room to fit a word in
                        a[index] = (sorted_ltp[k][0] + txt_height, a[index][1])
                        text_positions[index] = sorted_ltp[k][0] + txt_height
                        break

    return text_positions


# also from above source
def text_plotter(
    x_data, y_data, label_data, color_data, text_positions, axis, txt_widths, txt_height
):
    for x, y, label, color, t, width in zip(
        x_data, y_data, label_data, color_data, text_positions, txt_widths
    ):
        text = axis.text(
            x - width / 2, 0.5 * txt_height + t, f"{label}", rotation=0, color=color
        )
        # text.set_size(text.get_size() * 0.75)
        text.set_path_effects(
            [
                path_effects.PathPatchEffect(
                    edgecolor="white", linewidth=0.2, facecolor=color
                )
            ]
        )
        if 1 == 1:  # y != t:
            axis.vlines(
                x,
                y,
                0.4 * txt_height + t,
                linewidth=0.1,
                colors=color,
                linestyles="solid",
            )


def plot_for_exckel(args):
    # input: <Conv_FOsc> <Filtered_Peaks> <Spectrum_Peaks>
    # needs to be fed to:
    #    plot_spectra(nm_grid, oscillator_dist, epsilon_dist, excited_states, args)

    # <Conv_FOsc> contains the broadened spectrum in x y
    conv_fosc = np.loadtxt(args.exckel[0])
    eV_grid, oscillator_dist = conv_fosc.T

    eV2nm = 1239.841_973_862_09
    prefactor = 1.306_297_4
    standard_dev_eV = args.half_width_at_half_maximum / np.sqrt(np.log(2))

    # calculates the epsilon distribution
    epsilon_dist = 10 * prefactor * eV2nm * oscillator_dist / standard_dev_eV

    # <Filtered_Peaks> contains the filtered peaks in x y z with z being the labels
    filtered_peaks = np.loadtxt(args.exckel[1])
    filtered_eV, filtered_fosc, filtered_labels = filtered_peaks.T
    filtered_labels = filtered_labels.astype(int)

    # <Spectrum_Peaks> contains all peaks in x y z with z being the labels
    spectrum_peaks = np.loadtxt(args.exckel[2])
    peaks_eV, peaks_fosc, peaks_labels = spectrum_peaks.T
    peaks_labels = peaks_labels.astype(int)

    # initialize the plot
    fig, axs_f = plt.subplots(figsize=(6.75, 5), dpi=args.dots_per_inch)

    # grid
    if args.exckel_grid:
        plt.grid(b=True, which="major", axis="both")

    # plot each spectrum of the read files
    axs_f.plot(eV_grid, oscillator_dist, args.exckel_color)

    # set the ranges and labels
    axs_f.set_xlim(np.min(eV_grid), np.max(eV_grid))
    axs_f.set_ylim(0, 1.05 * np.max(oscillator_dist))
    axs_f.set_xlabel("Excitation energy / eV")
    axs_f.set_ylabel("Oscillator Strength $f$")

    # create the resp. stick spectra for each spectrum
    stem_lines = axs_f.stem(peaks_eV, peaks_fosc, markerfmt=" ", basefmt=" ")
    plt.setp(stem_lines, "color", args.exckel_color)

    # create the labels
    if args.peak_labels:
        xs = []
        ys = []
        label = []
        label_colors = []

        for peak in spectrum_peaks:
            nr = peak[2]
            eV = peak[0]
            fosc = peak[1]
            if nr in filtered_labels:
                xs.append(eV)
                ys.append(fosc)
                if args.peak_prefixes == []:
                    label.append(int(nr))
                else:
                    label.append(args.peak_prefixes[0] + "$_{" + str(int(nr)) + "}$")
                label_colors.append("k")

        # from https://stackoverflow.com/a/10739207
        x_data = xs
        y_data = ys

        # set the bbox for the text. Increase txt_width for wider text.
        my_renderer = aT.get_renderer(fig)
        textvar = plt.text(x_data[0], y_data[0], label[0])
        bla = aT.get_bboxes([textvar], my_renderer, (1, 1), ax=axs_f)
        tmp, txt_height = np.diff(bla[0], axis=0)[0]  # * 0.75
        textvar.remove()
        # txt_height = 0.04 * (plt.ylim()[1] - plt.ylim()[0])
        # txt_width = 0.02 * (plt.xlim()[1] - plt.xlim()[0])

        text_widths = []
        for x, y, lab in zip(xs, ys, label):
            textvar = plt.text(x, y, lab)
            bla = aT.get_bboxes([textvar], my_renderer, (1, 1), ax=axs_f)
            text_width, tmp = np.diff(bla[0], axis=0)[0]
            text_widths.append(text_width)  # * 0.75)
            textvar.remove()

        # get the corrected text positions, then write the text.
        text_positions = get_text_positions(x_data, y_data, text_widths, txt_height)
        text_plotter(
            x_data,
            y_data,
            label,
            label_colors,
            text_positions,
            axs_f,
            text_widths,
            txt_height,
        )

        if 1.05 * np.max(oscillator_dist) < max(text_positions) + 2 * txt_height:
            axs_f.set_ylim(0, max(text_positions) + 2 * txt_height)

    # top axis in nm ... most probably possible in an easier way
    #             ^^ I stick with my eV nomenclature, but it is nm now
    #                ... inverse to my spectra :)
    if args.top_eV:
        # copy nm-axis
        axs_top = axs_f.twiny()
        # create eV value list
        positions_eV = np.arange(0.0, 100.1, 1.0)  # eV
        # select only nm values in the plotted range
        positions_eV = np.array(
            [eV for eV in positions_eV if eV > np.min(eV_grid) and eV < np.max(eV_grid)]
        )
        # convert them to nm
        positions_nm = np.around(1239.841_973_862_09 / positions_eV, decimals=2)
        # reformats the nm tick positions to be between (nm_min, 0) and (nm_max, 1)
        positions_eV = (positions_eV - np.min(eV_grid)) / (
            np.max(eV_grid) - np.min(eV_grid)
        )
        # sets the tick positions
        axs_top.set_xticks(positions_eV)
        # sets the values at these tick positions
        axs_top.set_xticklabels(positions_nm)
        axs_top.set_xlabel("Wavelength $\lambda$ / nm")

    # create a second y axis on the right
    axs_eps = axs_f.twinx()
    axs_eps.set_ylim(0, 1.05 * np.max(epsilon_dist) / 10 ** 4)
    axs_eps.set_ylabel("Absorption $\epsilon$ / 10$^4$ L mol$^{-1}$ cm$^{-1}$")

    if not args.no_save:
        plt.savefig(f"{args.spectrum_out}", format=args.spectrum_out.split(".")[-1])

    if not args.no_plot:
        plt.show()


def plot_spectra(nm_grid, oscillator_dist, epsilon_dist, excited_states, args):
    # xkcd_colors = list(mcd.XKCD_COLORS.values())
    mpl_colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    if args.colors:
        mpl_colors = args.colors

    # initialize the plot
    fig, axs_f = plt.subplots(figsize=(6.75, 6.75 / args.aspect_ratio), dpi=args.dots_per_inch) # figsize was 6.75 at 5 before
    # plt.rcParams.update({"font.size": 14})

    # if a legend is wanted, either custom legend entries
    # or simply the filenames are printed
    if not args.no_legend:
        if args.plot_legend:
            labels = args.plot_legend
        else:
            labels = []
            if args.orca_out:
                labels += args.orca_out
            if args.orca_xy:
                labels += args.orca_xy
            if args.orca_soc:
                labels += args.orca_soc
            if args.orca5_soc:
                labels += args.orca5_soc
            if args.gaussian_out:
                labels += args.gaussian_out
            if args.adf_out:
                labels += args.adf_out
            if args.adf_soc:
                labels += args.adf_soc
            if args.exp:
                labels += args.exp

    # plot each spectrum of the read files
    cnt = 0
    for dist in oscillator_dist:
        color = mpl_colors[cnt]  # xkcd_colors[cnt]
        if args.no_legend:
            if args.linestyles != []:
                axs_f.plot(nm_grid[0], dist, color, linestyle=args.linestyles[cnt])
            else:
                axs_f.plot(nm_grid[0], dist, color)
        else:
            if args.linestyles != []:
                axs_f.plot(nm_grid[0], dist, color, label=labels[cnt], linestyle=args.linestyles[cnt])
            else:
                axs_f.plot(nm_grid[0], dist, color, label=labels[cnt])
        cnt += 1

    # set the ranges and labels
    axs_f.set_xlim(np.min(nm_grid[0]), np.max(nm_grid[0]))
    axs_f.set_ylim(0, 1.05 * np.max(oscillator_dist))
    axs_f.set_xlabel("Wavelength $\lambda$ / nm")
    axs_f.set_ylabel("Oscillator Strength $f$")

    # create the resp. stick spectra for each spectrum
    cnt = 0
    for states in excited_states:
        color = mpl_colors[cnt]  # xkcd_colors[cnt]
        cnt += 1
        nm_list = states[:, 2]
        oscillator_strengths = states[:, 3]
        stem_lines = axs_f.stem(
            nm_list, oscillator_strengths, markerfmt=" ", basefmt=" "
        )
        plt.setp(stem_lines, "color", color)

    # create the peak labels
    if args.peak_labels:
        texts = []
        xs = []
        ys = []
        label = []
        label_colors = []
        cnt = 0

        if args.label_nm_min != None:
            label_min = args.label_nm_min
        else:
            label_min = 1.1 * args.nm_min

        if args.label_nm_max != None:
            label_max = args.label_nm_max
        else:
            label_max = 0.9 * args.nm_max

        # otherwise it could be that the label_min > label_max, which is bs
        label_min, label_max = np.sort([label_min, label_max])

        for states in excited_states:
            for state in states:
                nr = state[0]
                nm = state[2]
                fosc = state[3]
                if fosc > args.fosc_threshold and nm >= label_min and nm <= label_max:
                    # print(f"state {nr:.0f}: {fosc} @ {nm} nm")
                    xs.append(nm)
                    ys.append(fosc)
                    if args.peak_prefixes == []:
                        label.append(int(nr))
                    else:
                        label.append(
                            args.peak_prefixes[cnt] + "$_{" + str(int(nr)) + "}$"
                        )
                    label_colors.append(mpl_colors[cnt])
            cnt += 1

        # from https://stackoverflow.com/a/10739207
        x_data = xs
        y_data = ys

        # set the bbox for the text. Increase txt_width for wider text.
        my_renderer = aT.get_renderer(fig)

        # show an error, if no peaks are in a label range
        # ... one could continue without the labeling and print a warning instead ...
        try:
            textvar = plt.text(x_data[0], y_data[0], label[0])
        except IndexError:
            sys.exit("No Peaks to label. Lower Threshold with -ft?")

        bla = aT.get_bboxes([textvar], my_renderer, (1, 1), ax=axs_f)
        tmp, txt_height = np.diff(bla[0], axis=0)[0]  # * 0.75
        textvar.remove()
        # txt_height = 0.04 * (plt.ylim()[1] - plt.ylim()[0])
        # txt_width = 0.02 * (plt.xlim()[1] - plt.xlim()[0])

        text_widths = []
        for x, y, lab in zip(xs, ys, label):
            textvar = plt.text(x, y, lab)
            bla = aT.get_bboxes([textvar], my_renderer, (1, 1), ax=axs_f)
            text_width, tmp = np.diff(bla[0], axis=0)[0]
            text_widths.append(text_width)  # * 0.75)
            textvar.remove()

        # get the corrected text positions, then write the text.
        text_positions = get_text_positions(x_data, y_data, text_widths, txt_height)
        text_plotter(
            x_data,
            y_data,
            label,
            label_colors,
            text_positions,
            axs_f,
            text_widths,
            txt_height,
        )

        # reset left y axis height if the labels would be truncated
        if 1.05 * np.max(oscillator_dist) < max(text_positions) + 2 * txt_height:
            axs_f.set_ylim(0, max(text_positions) + 2 * txt_height)

    # top axis in eV ... most probably possible in an easier way
    if args.top_eV:
        # copy nm-axis
        axs_top = axs_f.twiny()
        # create eV value list
        positions_eV = np.arange(0.5, 100.1, 0.5)  # eV
        # convert them to nm
        positions_nm = 1239.841_973_862_09 / positions_eV
        # select only nm values in the plotted range
        positions_nm = np.array(
            [
                nm
                for nm in positions_nm
                if nm > np.min(nm_grid[0]) and nm < np.max(nm_grid[0])
            ]
        )
        # converts these values back to eV
        positions_eV = np.around(1239.841_973_862_09 / positions_nm, decimals=1)
        # reformats the nm tick positions to be between (nm_min, 0) and (nm_max, 1)
        positions_nm = (positions_nm - np.min(nm_grid[0])) / (
            np.max(nm_grid[0]) - np.min(nm_grid[0])
        )
        # sets the tick positions
        axs_top.set_xticks(positions_nm)
        # sets the values at these tick positions
        axs_top.set_xticklabels(positions_eV)
        axs_top.set_xlabel("Excitation energy / eV")

    # change the height of the y axis to user defined value
    if args.y_height > 0.0:
        # calculate resize factor wrt to original value
        _, y_height = plt.ylim()
        y_resize_factor = args.y_height / y_height
        # reset y_height for f axis to user defined value
        axs_f.set_ylim(0, args.y_height)
    else:
        y_resize_factor = 1.0

    # create a secondary y axis on the right
    axs_eps = axs_f.twinx()
    axs_eps.set_ylim(0, y_resize_factor * 1.05 * np.max(epsilon_dist) / 10**3)
    axs_eps.set_ylabel("Extinction coefficient $\\epsilon$ / 10$^3$ L mol$^{-1}$ cm$^{-1}$")

    # create a tertiary y axis for experimental spectrum
    if args.tertiary_axis and args.exp:
        axs_eps2 = axs_f.twinx()
        axs_eps2.spines.right.set_position(("axes", 1.15))

    axs_exp = axs_eps
    if args.tertiary_axis and args.exp:
        axs_exp = axs_eps2

    if args.y2_height > 0.0 and args.exp:
        axs_eps.set_ylabel("Calc. extinction coefficient $\\epsilon$ / 10$^3$ L mol$^{-1}$ cm$^{-1}$")
        axs_exp.set_ylabel("Exp. extinctionc coefficient $\\epsilon$ / 10$^3$ L mol$^{-1}$ cm$^{-1}$")
        axs_exp.set_ylim(0, args.y2_height)

    # plot experimental spectra
    if args.exp:
        for file in args.exp:
            data = np.loadtxt(file).T
            if args.no_legend:
                axs_exp.plot(data[0], data[1] / 10 ** 3, "k--", alpha=0.5)
            else:
                axs_exp.plot(
                    data[0], data[1] / 10 ** 3, "k--", alpha=0.5, label=labels[cnt]
                )
            cnt += 1

    if not args.no_legend:
        f_lines, f_labels = axs_f.get_legend_handles_labels()
        # eps_lines, eps_labels = axs_eps.get_legend_handles_labels()
        eps_lines, eps_labels = axs_exp.get_legend_handles_labels()
        axs_f.legend(f_lines + eps_lines, f_labels + eps_labels)

    plt.tight_layout()

    if not args.no_save:
        plt.savefig(f"{args.spectrum_out}", format=args.spectrum_out.split(".")[-1])

    if not args.no_plot:
        plt.show()


def pgfdraw(arg):
    """yolo"""
    myoutput = f"\draw[blue](axis cs: {arg[0]},0) -- (axis cs: {arg[0]}, {arg[1]});"
    return myoutput


if __name__ == "__main__":
    main()
