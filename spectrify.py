#!/usr/bin/env python
import sys
import math
import argparse


def getinput(args):
    """parse the input"""
    parser = argparse.ArgumentParser(description=('G09 to UV-Vis-spectrum'
                                     ' converter.'))
    parser.add_argument('outputfile', metavar='G09-Output',
                        help=('Typically *.log or *.out, '
                              'but ending doesn\'t matter.'))
    parser.add_argument('--hwhh', '-w', default=0.333, type=float,
                        help=('half width at half height in eV '
                              '(default: 0.333)'))
    parser.add_argument('--xmin', '-i', default=280.0, type=float,
                        help=('minimal wavelength in nm '
                              '(default: 280.0)'))
    parser.add_argument('--xmax', '-f', default=780.0, type=float,
                        help=('maximal wavelength in nm '
                              '(default: 780.0'))
    parser.add_argument('--dx', '-s', default=1.0, type=float,
                        help=('stepsize of the spectrum in nm'
                              '(default: 1.0)'))
    parser.add_argument('--out', '-o', default='spectrum.dat',
                        help=('outputfile for the spectrum data '
                              '(default: spectrum.dat)'))
    parser.add_argument('--f4pgf', '-p', action='store_true',
                        help='\draw...-cmds for pgfplots\' osc. strengths')
    parser.add_argument('--fthresh', '-t', default=0.0, type=float,
                        help='threshold for oscillator strengths in --f4pgf')
    s2_group = parser.add_mutually_exclusive_group(required=False)
    s2_group.add_argument('--s2thresh', '-c', type=float,
                          help='threshold for <S**2> states ')
    s2_group.add_argument('--s2mult', '-m', type=int,
                          help='takes multiplicity to set <S**2>-threshold')
    parser.add_argument('--xylist', '-xy', action='store_true',
                        help='takes a simple nm osc-str list')

    return parser.parse_args(args)


def epsi(l, osz, hwhh, x):
    """Calculates epsilon(lambda) in L/(mol cm)"""
    s = 1239.84197386209 / hwhh
    eps1 = 1.3062974 * 10**8 * osz / (10**7 / s)
    eps2 = math.exp(-1.0 * ((1.0/x - 1.0/l) / (1.0 / s))**2)
    eps = eps1 * eps2
    return eps


def spec(lines, hwhh, x):
    """Calculates sum of eps at wavelength x"""
    y = 0.0
    for line in lines:
        l, osz, s2 = line
        y += epsi(float(l), float(osz), float(hwhh), x)
    return y


def pgfdraw(arg):
    """yolo"""
    myoutput = '\draw[blue](axis cs: '+str(arg[0])+',0) -- (axis cs:'
    myoutput += str(arg[0])+','+str(arg[1])+');'
    return myoutput


if __name__ == '__main__':
    args = getinput(sys.argv[1:])
    filename = args.outputfile
    hwhh = args.hwhh
    xi = args.xmin      # das ist x_initial
    xf = args.xmax      # das ist x_final
    stepsize = args.dx  # das ist die schrittweite
    outfile = args.out
    if args.s2thresh:
        ssqtrs = args.s2thresh
    elif args.s2mult:
        mult = args.s2mult
        s = 0.5*(mult-1.0)
        ssqtrs = s*(s+1.0)+0.75
    else:
        ssqtrs = 10.0**7

    if args.xylist:
        with open(sys.argv[1]) as f:
            nstates = sum(1 for _ in f)
    else:
        with open(sys.argv[1]) as f:
            nstates = sum("Excited State" in line for line in f)

    with open(sys.argv[1]) as f:
            lines = f.readlines()

    states = []

    i = 0
    if args.xylist:
        for line in lines:
            newline = line.split()
            states.append([float(newline[0]),
                           float(newline[1]),
                           0.0])
            i += 1
    else:
        for line in lines:
            if "Excited State" in line:
                newline = line.replace("f=", "")
                newline2 = newline.replace("<S**2>=", "")
                newline3 = newline2.split()
                if float(newline3[9]) < ssqtrs:
                    states.append([float(newline3[6]),
                                   float(newline3[8]),
                                   float(newline3[9])])
                    i += 1

    x = xi
    stringtowrite = "icm\teV\tnm\teps\tf\n"
    while x < xf+stepsize:
        y = spec(states, hwhh, x)
        stringtowrite += str(10**7/x)+"\t"              # cm**-1
        stringtowrite += str(1239.84197386209/x)+"\t"   # eV
        stringtowrite += str(x)+"\t"                    # nm
        stringtowrite += str(y)+"\t"                    # eps in L/(mol cm)
        stringtowrite += str(y/40490.05867167295)+"\n"  # osc. str. scale
        x += stepsize

    with open(outfile, 'w') as file:
        file.write(stringtowrite)

    if args.f4pgf:
        writepgf = ''
        for state in states:
            if state[1] >= args.fthresh:
                writepgf += pgfdraw(state)+"\n"

        with open('pgfdraw.dat', 'w') as file:
            file.write(writepgf)
