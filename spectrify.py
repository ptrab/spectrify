#!/usr/bin/env python
import sys
import argparse
import numpy as np

# import matplotlib
import matplotlib.colors as mcol
import matplotlib.pyplot as plt

# import matplotlib._color_data as mcd


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
        "--out",
        "-o",
        default="spectrum.dat",
        help=("outputfile for the spectrum data " "(default: spectrum.dat)"),
    )
    parser.add_argument(
        "--no-save", "-nos", action="store_true", help=("to store the plot or not")
    )
    # parser.add_argument(
    #     "--f4pgf",
    #     "-p",
    #     action="store_true",
    #     help="\draw...-cmds for pgfplots' osc. strengths",
    # )
    # parser.add_argument(
    #     "--fthresh",
    #     "-t",
    #     default=0.0,
    #     type=float,
    #     help="threshold for oscillator strengths in --f4pgf",
    # )
    # s2_group = parser.add_mutually_exclusive_group(required=False)
    # s2_group.add_argument(
    #     "--s2thresh", "-c", type=float, help="threshold for <S**2> states "
    # )
    # s2_group.add_argument(
    #     "--s2mult", "-m", type=int, help="takes multiplicity to set <S**2>-threshold"
    # )
    orca_group = parser.add_mutually_exclusive_group(required=False)
    orca_group.add_argument(
        "--orca-xy", "-oxy", nargs="*", help="takes a simple nm osc-str list"
    )
    orca_group.add_argument(
        "--print-s-squared",
        "-s2",
        action="store_true",
        help="prints a table for gaussian files with the spin contamination",
    )
    parser.add_argument("--orca-out", "-oout", nargs="*", help="orca output files")
    legend_group = parser.add_mutually_exclusive_group(required=False)
    legend_group.add_argument(
        "--plot-legend",
        "-leg",
        nargs="*",
        help="plot legend entries, orca xy first, then gaussian out",
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

    return parser.parse_args(args)


def main():
    args = getinput(sys.argv[1:])

    excited_states = []
    if args.orca_xy:
        for file in args.orca_xy:
            excited_states.append(get_orca_xy_excited_states(file))
    if args.orca_out:
        for file in args.orca_out:
            excited_states.append(get_orca_excited_states(file))
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

    if not args.orca_xy:
        cnt = 0
        if args.print_s_squared:
            for state_file in excited_states:
                print(f"file {cnt}")
                for state in state_file:
                    add_s2 = spin_contamination_additive(1, state[-1])
                    mix_s2 = spin_contamination_mixed(1, state[-1])
                    add_s2 = np.around(100 * add_s2, decimals=2)
                    mix_s2 = np.around(100 * mix_s2, decimals=2)
                    print(f"{state} {add_s2} {mix_s2}")

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
    fractions = np.array([1 / grid_point - 1 / nm_grid for grid_point in nm_list])
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
    # lying excited state has the greatest impact
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
    """ gives the contribution of the next hugher contaminating state,
    but is based on a more mixing formula than the additive one above
    (based on a discussion with Kevin Fiederling) """
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
    fig, axs_f = plt.subplots()

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

    # create a second y axis on the right
    axs_eps = axs_f.twinx()
    axs_eps.set_ylim(0, 1.05 * np.max(epsilon_dist) / 10 ** 4)
    axs_eps.set_ylabel("Absorption $\epsilon$ / 10$^4$ L mol$^{-1}$ cm$^{-1}$")

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

    axs_f.legend()
    fig.tight_layout()

    if not args.no_save:
        plt.savefig("spectra.svg", format="svg")
    plt.show()


def pgfdraw(arg):
    """yolo"""
    myoutput = f"\draw[blue](axis cs: {arg[0]},0) -- (axis cs: {arg[0]}, {arg[1]});"
    return myoutput


# https://stackoverflow.com/a/44960748/6155796
def wavelength_to_rgb(wavelength, gamma=0.8):
    """ taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).

    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    Additionally alpha value set to 0.5 outside range
    """
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 750:
        A = 1.0
    else:
        A = 0.5
    if wavelength < 380:
        wavelength = 380.0
    if wavelength > 750:
        wavelength = 750.0
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B, A)


###
# the follow belongs to the above conversion function
###
# clim=(350,780)
# norm = plt.Normalize(*clim)
# wl = np.arange(clim[0],clim[1]+1,2)
# colorlist = list(zip(norm(wl),[wavelength_to_rgb(w) for w in wl]))
# spectralmap = mcol.LinearSegmentedColormap.from_list("spectrum", colorlist)
#
# fig, axs = plt.subplots(1, 1, figsize=(8,4), tight_layout=True)
#
# wavelengths = np.linspace(200, 1000, 1000)
# spectrum = (5 + np.sin(wavelengths*0.1)**2) * np.exp(-0.00002*(wavelengths-600)**2)
# plt.plot(wavelengths, spectrum, color='darkred')
#
# y = np.linspace(0, 6, 100)
# X,Y = np.meshgrid(wavelengths, y)
#
# extent=(np.min(wavelengths), np.max(wavelengths), np.min(y), np.max(y))
#
# plt.imshow(X, clim=clim,  extent=extent, cmap=spectralmap, aspect='auto')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('Intensity')
#
# plt.fill_between(wavelengths, spectrum, 8, color='w')
# plt.savefig('WavelengthColors.png', dpi=200)
#
# plt.show()
###

if __name__ == "__main__":
    main()
