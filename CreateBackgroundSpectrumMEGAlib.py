#!/usr/bin/env python

""" Create a .dat file containing the spectrum of the primary
    and secondary protons plus albedo neutrons from the class
    LEOBackgroundGenerator to be used as input for the Step1
    *.source for Activation Simulations with MEGAlib.
"""

import numpy as np
import pandas as pd

from scipy.integrate import quad

import argparse

from LEOBackgroundGenerator import LEOBackgroundGenerator as LEO

# Instantiate the parser
pars = argparse.ArgumentParser(description='Create a .dat file containing '
                               + 'the spectrum of the primary and secondary '
                               + 'protons plus albedo neutrons from the class '
                               + 'LEOBackgroundGenerator to be used as input '
                               + 'for the Step1 *.source for Activation '
                               + 'Simulations with MEGAlib.')

pars.add_argument('-i', '--inclination', type=float, nargs='?',
                  default=0., help='Inclination of the orbit in degree [0.]')

pars.add_argument('-a', '--altitude', type=float, nargs='?',
                  default=550., help='Altitude of the orbit in km [550.]')

pars.add_argument('-el', '--elow', type=float, nargs='?',
                  default=1, help='Log10 of the lowest energy limit in keV [1]')

pars.add_argument('-eh', '--ehigh', type=float, nargs='?',
                  default=8, help='Log10 of the highest energy limit in keV [8]')

args = pars.parse_args()

Inclination = args.inclination
Altitude = args.altitude

Elow = args.elow
Ehigh = args.ehigh

LEOClass = LEO(1.0*Altitude, 1.0*Inclination)

ViewAtmo = 2*np.pi * (np.cos(np.deg2rad(LEOClass.HorizonAngle)) + 1)
ViewSky = 2*np.pi * (1-np.cos(np.deg2rad(LEOClass.HorizonAngle)))

Particle = ["AtmosphericNeutrons", "PrimaryProtons", "SecondaryProtonsUpward",
            "SecondaryProtonsDownward", "PrimaryAlphas", "CosmicPhotons", "AlbedoPhotons"]

Megalibfunc = [LEOClass.AtmosphericNeutrons, LEOClass.PrimaryProtons,
               LEOClass.SecondaryProtonsUpward, LEOClass.SecondaryProtonsDownward,
               LEOClass.PrimaryAlphas, LEOClass.CosmicPhotons, LEOClass.AlbedoPhotons]

fac = [ViewAtmo, ViewSky, 2*np.pi, 2*np.pi, ViewSky, ViewSky, ViewAtmo]

for i in range(0, len(Megalibfunc)):

    Energies = np.logspace(Elow, Ehigh, num=100, endpoint=True, base=10.0)
    Output = "%s_Spec_%skm_%sdeg.dat" % (Particle[i], int(Altitude), int(Inclination))
    IntSpectrum, err = quad(Megalibfunc[i], 10**Elow, 10**Ehigh)
    print(Particle[i], IntSpectrum*fac[i], " #/cm^2/s", err)
    with open(Output, 'w') as f:
        print('# %s spectrum ' % Particle[i], file=f)
        print('# Format: DP <energy in keV> <shape of differential spectrum [XX/keV]>', file=f)
        print('# Although cosima doesn\'t use it the spectrum here is given as a flux in #/cm^2/s/keV', file=f)
        print('# Integrated over %s sr' % fac[i], file=f)
        print('# Integral Flux: %s #/cm^2/s' % (IntSpectrum*fac[i]), file=f)
        print('', file=f)
        print('IP LOGLOG', file=f)
        print('', file=f)
        for j in range(0, len(Energies)):
            print('DP', Energies[j], Megalibfunc[i](Energies[j]), file=f)
        print('EN', file=f)
