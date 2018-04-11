#!/usr/bin/env python

"""
    Macro to generate a background spectrum for the Galactic
    Center/Disk region starting from Fermi-LAT public data:
    https://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gll_iem_v06.fits
    It creates a file (OutputFile) containing the values used by the class
    LEOBackgroundGenerator
"""

__author__ = 'Paolo Cumani, Vincent Tatischeff'


from astropy.io import fits
import numpy as np

from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord


OutputFile = 'LATBackground.dat'

NdegreeX1 = 5
NdegreeX2 = 90-NdegreeX1/2.
NdegreeY = 2

try:
    back = fits.open("./Data/gll_iem_v06.fits")
except IOError as err:
    print("\nFile ./Data/gll_iem_v06.fits not found: you can download it "
          "at https://fermi.gsfc.nasa.gov/ssc/data/analysis/software"
          "/aux/gll_iem_v06.fits\n\n")
    raise

image = back[0].data
Energy = back[1].data
wcs = WCS(back[0].header)
convx = back[0].header['CDELT1']
convy = back[0].header['CDELT2']

position = SkyCoord(frame="galactic", l=[0], b=[0], unit="deg")

sizeGC = (NdegreeY/convy, NdegreeX1/convx)
sizeDisk = (NdegreeY/convy, NdegreeX2/convx)

with open(OutputFile, 'w') as f:
    print('# Energy/keV FluxGCAv FluxGCMax FluxDiskAv FluxDiskMax / '
          '#/cm^2/s/sr/keV', file=f)
    print('Energy FluxGCAv FluxGCMax FluxDiskAv FluxDiskMax', file=f)

    for i in range(0, len(Energy)):
        cutoutGC = Cutout2D(image[i, :, :], position, sizeGC,
                            wcs=wcs.dropaxis(2))
        MeanFluxGC = np.average(cutoutGC.data)/10**3
        position1 = SkyCoord(frame="galactic",
                             l=[(NdegreeX2+NdegreeX1)/2.], b=[0],
                             unit="deg")
        position2 = SkyCoord(frame="galactic",
                             l=[-(NdegreeX2+NdegreeX1)/2.], b=[0],
                             unit="deg")

        cutoutDisk1 = Cutout2D(image[i, :, :], position1, sizeDisk,
                               wcs=wcs.dropaxis(2))
        cutoutDisk2 = Cutout2D(image[i, :, :], position2, sizeDisk,
                               wcs=wcs.dropaxis(2))

        diskarray = np.concatenate((cutoutDisk1.data,
                                    cutoutDisk2.data))

        MeanFluxDisk = np.average(diskarray)/10**3
        print("%s %s %s %s %s" % ("".join(str(10**3*float(x))
              for x in Energy[i]), MeanFluxGC,
              np.max(cutoutGC.data)/10**3, MeanFluxDisk,
              np.max(diskarray)/10**3), file=f)
