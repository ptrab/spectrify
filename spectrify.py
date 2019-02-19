#!/usr/bin/env python3
import sys
import argparse
import numpy as np

import adjustText as aT

import matplotlib.patheffects as path_effects
import matplotlib.colors as mcol
import matplotlib.pyplot as plt

# import matplotlib._color_data as mcd

from scipy import interpolate


def getinput(args):
    """parse the input"""
    parser = argparse.ArgumentParser(
        description=("G09 to UV-Vis-spectrum" " converter.")
    )
    parser.add_argument(
        "--gaussian-out",
        "-gout",
        metavar="G09-Output",
        nargs="*",
        help=("Typically *.log or *.out, " "but ending doesn't matter."),
    )
    parser.add_argument(
        "--hwhh",
        "-w",
        default=0.333,
        type=float,
        help=("half width at half height in eV " "(default: 0.333)"),
    )
    parser.add_argument(
        "--xmin",
        "-i",
        default=280.0,
        type=float,
        help=("minimal wavelength in nm " "(default: 280.0)"),
    )
    parser.add_argument(
        "--xmax",
        "-f",
        default=780.0,
        type=float,
        help=("maximal wavelength in nm " "(default: 780.0"),
    )
    parser.add_argument(
        "--dx",
        "-s",
        default=1.0,
        type=float,
        help=("stepsize of the spectrum in nm" "(default: 1.0)"),
    )
    parser.add_argument(
        "--spectrum-out",
        "--out",
        "-o",
        default="spectrum.dat",
        help=("outputfile for the spectrum data " "(default: spectrum.dat)"),
    )
    parser.add_argument(
        "--spectrum-file-extension",
        "-ext",
        "-end",
        default="svg",
        help="file extension for the saved spectrum",
    )
    parser.add_argument(
        "--no-save", "-nos", action="store_true", help=("to store the plot or not")
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
        "--orca-soc", "-osoc", nargs="*", help="orca output files but SOC results"
    )
    legend_group = parser.add_mutually_exclusive_group(required=False)
    legend_group.add_argument(
        "--plot-legend",
        "-leg",
        nargs="*",
        help="plot legend entries, orca out > orca xy > orca soc > gaussian out",
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
        help="add prefixes to the labels, orca out > orca xy > orca soc > gaussian out",
    )
    parser.add_argument(
        "--fosc-threshold",
        "-ft",
        type=float,
        default=0.05,
        help="sets the oscillator strength threshold for labeling",
    )

    return parser.parse_args(args)


def main():
    args = getinput(sys.argv[1:])

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
    if args.gaussian_out:
        for file in args.gaussian_out:
            excited_states.append(get_gaussian_excited_states(file))

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

    standard_dev_eV = args.hwhh / np.sqrt(np.log(2))
    step_size = args.dx

    nm_list = excited_states[:, 2]
    oscillator_strengths = excited_states[:, 3]

    nm_grid = np.arange(args.xmin, args.xmax + step_size / 2, step_size)
    oscillator_multipliers = oscillator_strengths
    # the 10 is the result of 10 ** 8 / 10 ** 7, from which the first was given in the
    # white paper as part of the prefactor and the latter comes from the conversion
    # between nm and cm^-1
    epsilon_multipliers = (
        10 * prefactor * eV2nm * oscillator_multipliers / standard_dev_eV
    )
    fractions = np.array([1.0 / grid_point - 1.0 / nm_grid for grid_point in nm_list])
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
    """ gives the contribution of the next higher contaminating state """
    # it assumes, that in spin contamination the next highest
    # lying excited state has the greatest impact and that the
    # mixing is additive, as with 100% CONTAMINATION one should
    # have both states and not only the contaminating one
    #
    # <S²> = <S²> + k * <(S+1)²>
    #
    #     <S²> - S * (S + 1)
    # k = ------------------
    #      (S + 1) * (S + 2)
    #
    result = (s_squared - spin * (spin + 1)) / ((spin + 1) * (spin + 2))
    return result


def spin_contamination_mixed(spin, s_squared):
    """ gives the contribution of the next higher contaminating state,
    but is based on a more mixing formula than the additive one above
    (based on a discussion with Kevin Fiederling)
    while the former equation was my interpretation, this seems to be
    in accordance with a Casida paper: 10.1016/j.theochem.2009.07.036
    """
    # it assumes, that in spin contamination the next highest
    # lying excited state has the greatest impact but also
    # that the mixing is not additive, as above
    #
    # <S²> = (1 - k) + <S²> + k <(S+1)²>
    #
    #     <S²> - S * (S - 1)
    # k = ------------------
    #        2 * (S + 1)
    #
    # this value is always larger than the above formula
    # by a factor of 1 + S / 2
    result = (s_squared - spin * (spin + 1)) / (2 * (spin + 1))
    return result


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

    # initialize the plot
    fig, axs_f = plt.subplots(figsize=(6.75, 5))  # , dpi=300)
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
            if args.gaussian_out:
                labels += args.gaussian_out

    # plot each spectrum of the read files
    cnt = 0
    for dist in oscillator_dist:
        color = mpl_colors[cnt]  # xkcd_colors[cnt]
        if args.no_legend:
            axs_f.plot(nm_grid[0], dist, color)
        else:
            axs_f.plot(nm_grid[0], dist, color, label=labels[cnt])
        cnt += 1

    # set the and ranges
    axs_f.set_xlim(np.min(nm_grid[0]), np.max(nm_grid[0]))
    axs_f.set_ylim(0, 1.05 * np.max(oscillator_dist))
    axs_f.set_xlabel("Wavelength / nm")
    axs_f.set_ylabel("Oscillator Strength f")

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

    # create the labels
    if args.peak_labels:
        texts = []
        xs = []
        ys = []
        label = []
        label_colors = []
        cnt = 0
        for states in excited_states:
            for state in states:
                nr = state[0]
                nm = state[2]
                fosc = state[3]
                if fosc > args.fosc_threshold and nm > args.xmin and nm < args.xmax:
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

    # create a second y axis on the right
    axs_eps = axs_f.twinx()
    axs_eps.set_ylim(0, 1.05 * np.max(epsilon_dist) / 10 ** 4)
    axs_eps.set_ylabel("Absorption $\epsilon$ / 10$^4$ L mol$^{-1}$ cm$^{-1}$")

    axs_f.legend()
    # fig.tight_layout()

    if not args.no_save:
        plt.savefig(
            f"spectra.{args.spectrum_file_extension}",
            format=args.spectrum_file_extension,
        )
    plt.show()


def pgfdraw(arg):
    """yolo"""
    myoutput = f"\draw[blue](axis cs: {arg[0]},0) -- (axis cs: {arg[0]}, {arg[1]});"
    return myoutput


if __name__ == "__main__":
    main()
