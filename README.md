sedstacker
==========

A Python toolkit for statistically combining multiple spectral energy distributions (SEDs).

** This version is unstable! **

Purpose
-------
sedstacker was developed for interest in creating SED template models, increasing the signal-to-noise ratio of faint spectra, and studying the average SED characteristics of specific groups of astronomical objects.

Installation
------------

More to come at a later date...

Intro
-----

Users can load SED data from file, shift the SEDs to some redshift, normalize the SEDs in one of several ways, and then stack the data together to create a statistically-combined SED.

    >>> from sedstacker import io
    >>> from sedstacker.sed import Sed, AggregateSed, stack
    >>>
    >>> sed1 = io.load_sed('data/sed1.txt', sed_type='sed')
    >>> sed1.z = 1.2
    >>> spec1 = io.load_sed('data/spec1.txt', sed_type='spectrum')
    >>> spec1.z = 0.54
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
    >>> high_lum_low_z_seds = io.load_cat('data/high_lum_low_z_seds.cat', column_map=column_map)

### sedstacker Classes

PhotometricPoint - A PhotometricPoint is defined by an x-y pair, and possibly a y-err. The x- and y-units may also be stored in the PhotometricPoint.

    >>> point = PhotometricPoint(4470.9, 3.56e-11, yerr=1.0e-13, xunit='AA', yunit='erg/s/cm**2/AA')

Segment - A collection of x-y pairs, with optional y-errors, x- and y-units, and a redshift, z. SEDs and spectra are considered types of segments. This class is an interface, and is **not** meant to be utilized by the user.

Sed - A set of PhotometricPoints. A Sed can represent the entire SED of one astronomical source or a portion of the source’s SED. Seds can have PhotometricPoints added, removed and masked.

Spectrum - A Segment object of a spectrum in the scientific sense; it is meant to represent data taken from a spectrometer.

AggregateSed - A collection of Segments (Spectra and/or Seds), usually from one astronomical source.

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

