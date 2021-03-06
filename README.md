<h1>Background on a low Earth orbit</h1>

Macros to compute and visualize the background for a satellite on a Low Earth Orbit (LEO). 
The model used here and results using these macros are presented in:

P. Cumani, M. Hernanz, J. Kiener, V. Tatischeff, and A. Zoglauer. "Background for a gamma-ray satellite on a low-Earth orbit". Feb 2019. arXiv:1902.06944, doi:10.1007/s10686-019-09624-0.

Please use this reference if you present any result obtain using these classes.

**Table of Contents**

* [Packages](#packages)
* [Description](#description)
* [Bibliography](#bibliography)
* [Validity limits](#limits)

<h2>Packages</h2>
The macros are written for Python3. It uses the following packages:

* Numpy
* Astropy
* Scipy
* Matplotlib
* Pandas

<h2>Description</h2>

* LEOBackgroundGenerator.py : contain the definition of the class describing all the background components.
* LATBackground.py : creates the file Data/LATBackground.dat from the Fermi fits file. It needs to be used only if there is a change in the Fermi-LAT background or a change in the areas where the flux is calculated (lines 24-26).
* BackgroundPlotter_All.py : plots all the different components. To be run with a -h (or --help) option to show a help.
* CreateBackgroundSpectrumMEGAlib.py : creates a file describing the spectrum of different components to be used with MEGAlib to define a source. The value of the integral flux, calculated using the appropriate solid angle and to be added to the .source file, is added as a comment inside the newly created file. To be run with a -h (or --help) option to show a help.

<h2>Bibliography</h2>
The model uses equations/data from:

* Albedo Neutrons: 
  - Kole et al. 2015\
&nbsp;&nbsp; doi:10.1016/j.astropartphys.2014.10.002
  - Lingenfelter 1963\
&nbsp;&nbsp; doi:10.1029/JZ068i020p05633
* Cosmic Photons: 
  - Türler et al. 2010\
&nbsp;&nbsp; doi:10.1051/0004-6361/200913072
  - Mizuno et al. 2004\
&nbsp;&nbsp; http://stacks.iop.org/0004-637X/614/i=2/a=1113
  - Ackermann et al. 2015\
&nbsp;&nbsp; doi:10.1088/0004-637X/799/1/86
* Galactic Center/Disk: 
  - Fermi-LAT collaboration\
&nbsp;&nbsp; https://fermi.gsfc.nasa.gov/ssc/data/analysis/software/aux/gll_iem_v06.fits
* Primary Protons: 
  - Aguilar et al. 2015\
&nbsp;&nbsp; doi:10.1103/PhysRevLett.114.171103
* Secondary Protons: 
  - Mizuno et al. 2004\
&nbsp;&nbsp; http://stacks.iop.org/0004-637X/614/i=2/a=1113
* Primary Alphas:  
  - Aguilar et al. 2015b\
&nbsp;&nbsp; doi:10.1103/PhysRevLett.115.211101
* Primary Electrons: 
  - Aguilar et al. 2014\
&nbsp;&nbsp; doi:10.1103/PhysRevLett.113.121102
  - Mizuno et al. 2004\
&nbsp;&nbsp; http://stacks.iop.org/0004-637X/614/i=2/a=1113
* Primary Positrons: 
  - Aguilar et al. 2014\
&nbsp;&nbsp; doi:10.1103/PhysRevLett.113.121102
  - Mizuno et al. 2004\
&nbsp;&nbsp; http://stacks.iop.org/0004-637X/614/i=2/a=1113
* Secondary Electrons: 
  - Mizuno et al. 2004\
&nbsp;&nbsp; http://stacks.iop.org/0004-637X/614/i=2/a=1113
* Secondary Positrons: 
  - Mizuno et al. 2004\
&nbsp;&nbsp;  http://stacks.iop.org/0004-637X/614/i=2/a=1113
* Atmospheric Photons: 
  - Sazonov et al. 2007\
&nbsp;&nbsp; doi:10.1111/j.1365-2966.2007.11746.x
  - Churazov et al. 2006\
&nbsp;&nbsp;  doi:10.1111/j.1365-2966.2008.12918.x
  - Türler et al. 2010\
&nbsp;&nbsp; doi:10.1051/0004-6361/200913072
  - Mizuno et al. 2004\
&nbsp;&nbsp; http://stacks.iop.org/0004-637X/614/i=2/a=1113
  - Abdo et al. 2009\
&nbsp;&nbsp; doi:10.1103/PhysRevD.80.122004

<h2>Validity limits</h2><a name="limits"/>

| | Orbit Parameters | Energy |
| :--- | :---: | ---:|
| Cosmic Photons | Independent | 4 keV - 820 GeV
| Galactic Photons | Independent | 58 MeV -  ~513 GeV
| Albedo Photons | All LEOs | 1 keV - 400 GeV
| Primary Protons | Altitude > 100 km | 10 MeV - 10 TeV
| Primary Alphas | Altitude > 100 km | 10 MeV - 10 TeV
| Primary Electrons/Positrons | Altitude > 100 km | 570 MeV - 429 GeV
| Secondary Protons | 1.06	&le; R<sub>cutoff</sub> &le; 12.47 | 1 MeV - 10 GeV
| Secondary Electrons/Positrons | 1.06	&le; R<sub>cutoff</sub> &le; 12.47 | 1 MeV - 20 GeV
| Atmospheric Neutrons | Altitude: ~100 km - ~1000 km / Inclination: < 65&deg;| 0.01 eV - ~30 GeV

Validity limits of the different components of the background.

<h2>Acknowledgment</h2>
This work has been carried out in the framework of the project AHEAD, funded by the European Union
