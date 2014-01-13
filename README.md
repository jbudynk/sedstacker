sedstacker
==========

A Python toolkit for statistically combining multiple spectral energy distributions (SEDs).

** This version is unstable! **

Documentation
-------------

Please view the Iris user documentation at [cxc.harvard.edu/iris/].

Installation
------------

More to come at a later date...

Quick guide
-----------

sedstacker was developed for interest in creating SED template models,
increasing the signal-to-noise ratio of faint spectra, and studying
the average SED characteristics of specific groups of astronomical
objects.

Users can load SED data from file, shift the SEDs to some redshift,
normalize the SEDs in one of several ways, and then stack the data
together to create a statistically-combined SED.

### Important Classes

PhotometricPoint - A PhotometricPoint is defined by an x-y pair, and possibly a y-err. The x- and y-units may also be stored in the PhotometricPoint.

    >>> point = PhotometricPoint(4470.9, 3.56e-11, yerr=1.0e-13, xunit='AA', yunit='erg/s/cm**2/AA')

Segment - A collection of x-y pairs, with optional y-errors, x- and
y-units, and a redshift, z. SEDs and spectra are considered types of
segments. This class is an interface, and is **not** meant to be
utilized by the user. 

Sed - A set of PhotometricPoints. A Sed can represent the entire SED
of one astronomical source or a portion of the source’s SED. Seds can
have PhotometricPoints added, removed and masked.

    >>> spec = [3823.0,4459.7,5483.8,4779.6,6295.1,7640.8]
    >>> flux = [20.43,20.28,20.18,20.18,22.73,21.01]
    >>> fluxerr = [0.08]*len(flux)
    >>> redshift = 0.31
    >>> sed = sedstacker.sed.Sed(x=spec, y=flux, yerr=fluxerr, xunit='AA', yunit='mag', z=redshift) 

Spectrum - A Segment object of a spectrum in the scientific sense; it
is meant to represent data taken from a spectrometer. 

AggregateSed - A collection of Segments (Spectra and/or Seds), usually
from one astronomical source. Users may inspect and manipulate each
Segment in the AggregateSed independently.

### File I/O

Users can load two types of datasets from file: spectra and photometry catalogs. Both file types are ASCII tables, meaning they're composed of white-spaced separated columns, with each column having the same number of rows.

A *spectrum file* is a file with at least  two white-space-separated columns: one with the spectral coordinates (in wavelength), the other with the corresponding flux coordinates. The file may also have a third column with the corresponding flux errors. Ex:

    #wavelength          flux     flux_err
    2266.1474     0.032388065  1.015389e-4
    2266.9024    0.0082257072  6.264234e-5
    2267.6575     0.026719441  1.910421e-4
    2268.4125     0.029819652  7.046351e-4
    2269.1676     0.092393999  1.372951e-4
    2269.9226    0.0045967875  2.348762e-5
    2270.6777     0.010004328  5.329874e-4

To load a spectrum file with sedstacker:

    >>> from sedstacker import io
    >>> spectrum = io.load_sed('path/to/file.dat', sed_type='spectrum')
    >>> print spectrum

A *photometry catalog* is a file with whitespace-separated columns, each with the same number of rows, where each row represents a unique astronomical object. The file must have at least two columns: one with the ID for each object, and one with fluxes for a specific spectral value. There can be any number of columns in a Photometry Catalog, but columns must have unique names. Ex:

    # ID   z   ucfht errU Bsub  errB Vsub  errV gsub  errg
    2051 0.668 20.43 0.09 20.28 0.09 20.18 0.09 20.18 0.09
    41   0.962 22.73 0.09 22.38 0.09 21.93 0.09 22.44 0.09
    164  0.529 21.16 0.09 21.05 0.09 20.84 0.09 21.01 0.09

### Common use cases

The user has a photometry catalog file of SEDs which she wants to
categorize by redshift before stacking.

    >>> from sedstacker.io import load_cat
    >>> from sedstacker.sed import AggregateSed, stack
    >>>
    >>> # create the mapping from column fluxes to spectral
    >>> # coordinates, spectral and flux units, and flux-error values
    >>> column_map={"ucfht":(3823.0,"AA","mag","errU"),
                    "Bsub":(4459.7,"AA","mag","errB"),
              	    "Vsub":(5483.8,"AA","mag","errV"),
              	    "gsub":(4779.6,"AA","mag","errg"),
              	    "rsub":(6295.1,"AA","mag","errR"),
              	    "isub":(7640.8,"AA","mag","errI"),
              	    "zsub":(9036.9,"AA","mag","errz"),
              	    "j":(12491.0,"AA","mag","errJ"),
              	    "Ks":(21590.4,"AA","mag","errK"),
              	    "irac3.6":(36000.0,"AA","mag","err3.6"),
              	    "irac4.5":(45000.0,"AA","mag","err4.5"),
              	    "irac5.8":(58000.0,"AA","mag","err5.8"),
              	    "irac8.0":(80000.0,"AA","mag","err8.0"),
              	    "mips24.0":(240000.0,"AA","mag","err24")
              	   }
    >>> seds = load_cat('mycatalog.cat', column_map=column_map)
    >>>
    >>> # categorize seds by their redshifts: sources with z < 0.5,
    >>> # sources with z > 0.5
    >>> lowz = AggregateSed([sed for sed in seds if sed.z < 0.5])
    >>> highz = AggregateSed([sed for sed in seds if sed.z >= 0.5])
    >>>
    >>> # shift the SEDs to restframe
    >>> lowz.shift(0.0)
    >>> highz.shift(0.0)
    >>>
    >>> # normalize the SEDs so that the integrated flux of each SED is
    >>> # 1
    >>> lowz.normalize_by_int()
    >>> highz.normalize_by_int()
    >>>
    >>> # stack the lowz SEDs using weighted average and logarithmic
    >>> # binning
    >>> lowz_stack = stack(lowz, 0.2, 'wavg', logbin=True)

Users can load data from file using the *io* module:

    >>> from sedstacker import io

Store data as a SED object, and set the SED's redshift

    >>> sed1 = io.load_sed('data/sed1.txt', sed_type='sed')
    >>> sed1.z = 1.2

Store data as a Spectrum object, and set the SED's redshift

    >>> spec1 = io.load_sed('data/spec1.txt', sed_type='spectrum')
    >>> spec1.z = 0.54

Load data from a photometry catalog, providing the necessary
header information to map columns to a spectral value,
units, and flux-error column.

    >>> column_map = {"ucfht":(3823.0,"AA","mag","errU"),
                      "Bsub":(4459.7,"AA","mag","errB"),
                      "Vsub":(5483.8,"AA","mag","errV"),
                      "gsub":(4779.6,"AA","mag","errg"),
                      "rsub":(6295.1,"AA","mag","errR"),
                      "isub":(7640.8,"AA","mag","errI"),
                      "zsub":(9036.9,"AA","mag","errz"),
                      "j":(12491.0,"AA","mag","errJ"),
                      "Ks":(21590.4,"AA","mag","errK"),
                      }
    >>> high_lum_low_z_seds = io.load_cat('data/high_lum_low_z_seds.cat',
                                          column_map=column_map)

Each row is stored as a SED object; the collection of SEDs are then stored
as an AggregateSed object (i.e. high_lum_low_z_seds is an
AggregateSed). The other column names in the file are stored as
attributes for each SED stored in the AggregateSed. In this case, the
redshift is in the data file, with column name "z":

    >>> print high_lum_low_z_seds.z
    [0.668, 0.962, 0.529, 0.9, 1.456, 2.415]

An AggregateSed is simply a list of SEDs and Spectra. Information
about the individual SEDs can be accessed as one would access data to
a Python list:




