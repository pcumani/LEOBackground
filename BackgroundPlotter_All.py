#!/usr/bin/env python


""" Plot the background components from the class LEOBackgroundGenerator
 And save it as a pdf
"""

__author__ = 'Paolo Cumani'

import matplotlib.pyplot as plt

import numpy as np
import argparse

from LEOBackgroundGenerator import LEOBackgroundGenerator as LEO


# Instantiate the parser
pars = argparse.ArgumentParser(description='Plot the background'
                                           + ' components from the class'
                                           + ' LEOBackgroundGenerator')

pars.add_argument('-i', '--inclination', type=float, nargs='?',
                  default=0., help='Inclination of the orbit in degree [0.]')

pars.add_argument('-a', '--altitude', type=float, nargs='?',
                  default=550., help='Altitude of the orbit in km [550.]')

pars.add_argument('-o', '--outputpdf', type=str, nargs='?',
                  default="FullSpectrum", help='Name of the output pdf [FullSpectrum]')

args = pars.parse_args()

Energies = np.logspace(1, 8, num=1000000, endpoint=True, base=10.0)

LeoBack = LEO(args.altitude, args.inclination)

LeoBackfunc = [LeoBack.AlbedoNeutrons, LeoBack.CosmicPhotons,
               LeoBack.PrimaryProtons, LeoBack.SecondaryProtons,
               LeoBack.PrimaryAlphas, LeoBack.PrimaryElectrons,
               LeoBack.PrimaryPositrons, LeoBack.SecondaryElectrons,
               LeoBack.SecondaryPositrons, LeoBack.AlbedoPhotons,
               LeoBack.GalacticCenter, LeoBack.GalacticDisk]

Title = ["Albedo Neutrons", "Cosmic Photons", "Primary Protons",
         "Secondary Protons", "Primary Alphas", "Primary Electrons",
         "Primary Positrons", "Secondary Electrons", "Secondary Positrons",
         "Albedo Photons", "Galactic Center", "Galactic Disk"]

colors = ['darkred', 'darkorange', 'darkgreen', 'steelblue', 'darkblue',
          'orchid', 'red', 'darkgrey', 'mediumseagreen', 'black',
          'gold', 'hotpink']

dash = [(5, 0), (5, 2), (5, 2, 1, 2), (2, 1), (5, 5), (5, 3, 3, 3),
        (5, 10), (5, 3, 3, 3), (5, 5, 5, 5), (3, 5),
        (9, 2), (7, 3)]

fig1, ax1 = plt.subplots(nrows=1, ncols=1, sharex=True, figsize=(12, 8))

plt.minorticks_on()

ax1.grid(color='lightgrey', which='major', linestyle='-', linewidth=1)
ax1.grid(color='lightgrey', which='minor', linestyle='--', linewidth=1)

for i in range(0, len(LeoBackfunc)):
    masknan = ~np.isnan(LeoBackfunc[i](Energies))
    ax1.loglog(Energies[masknan]/1000,
               10000*1000*LeoBackfunc[i](Energies)[masknan],
               color=colors[i], linestyle='--', dashes=dash[i], label=Title[i])

handles, labels = ax1.get_legend_handles_labels()

ax1.set_xlim([Energies[0]/1000, Energies[-1]/1000])
ax1.set_ylim([10**(-6), 10**7])

ax1.title.set_text("Background spectrum in a low Earth orbit (LEO)")
ax1.title.set_size(20)

ax1.set_ylabel(r'Flux / m$^{-2}$s$^{-1}$MeV$^{-1}$sr$^{-1}$',
               verticalalignment='bottom', labelpad=20, fontsize=15)

legend = ax1.legend(handles, labels, loc='upper right',
                    prop={'size': 15}, fancybox=True)
legend.get_frame().set_alpha(0.9)

ax1.set_xlabel(r'Energy / MeV', verticalalignment='center', labelpad=20, fontsize=15)

fig1.savefig(args.outputpdf+".pdf", bbox_inches='tight')
