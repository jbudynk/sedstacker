sedstacker
==========

A Python toolkit for statistically combining multiple spectral energy distributions (SEDs).

Installation
------------

From the command line, run

    git clone --recursive git://github.com/jbudynk/sedstacker.git

`cd` into `sedstacker` and run

    python setup.py install

Documentation
-------------

You can build the HTML documentation pages on your machine.

Install `sphinx` and `numpydoc`, `cd` into the `docs` directory, and then run `make html`

    $: pip install sphinx numpydoc
    $: cd <path-to>/sedstacker/docs
    $: make html

The HTML files will be stored in `_build/html`.

Quick guide
-----------

``sedstacker`` was developed for interest in creating SED template models,
increasing the signal-to-noise ratio of faint spectra, and studying
the average SED characteristics of specific groups of astronomical
objects.

Users can load SED data from file, shift the SEDs to some redshift,
normalize the SEDs in one of several ways, and then stack the data
together to create a statistically-combined SED.

### SED Data Classes

PhotometricPoint - A PhotometricPoint is defined by an x-y pair, and possibly a y-err.
The x- and y-units may also be stored in the PhotometricPoint.

    >>> point = PhotometricPoint(4470.9, 3.56e-11, yerr=1.0e-13, xunit='AA', yunit='erg/s/cm**2/AA')

Segment - A collection of x-y pairs, with optional y-errors, x- and
y-units, and a redshift, z. SEDs and spectra are considered types of
segments. This class is an interface, and is **not** meant to be
utilized by the user. 

Sed - A set of PhotometricPoints. A Sed can represent the entire SED
of one astronomical source or a portion of the source’s SED. Seds can
have PhotometricPoints added, removed and masked.

    >>> spec = [3823.0, 4459.7, 5483.8, 4779.6, 6295.1, 7640.8]
    >>> flux = [20.43, 20.28, 20.18, 20.18, 22.73, 21.01]
    >>> fluxerr = [0.08]*len(flux)
    >>> redshift = 0.31
    >>>
    >>> sed = sedstacker.sed.Sed(x=spec, y=flux, yerr=fluxerr, xunit=['AA'], yunit=['mag'], z=redshift)

Spectrum - A Segment object of a spectrum in the scientific sense; it
is meant to represent data taken from a spectrometer.

    >>> spectrum = sedstacker.sed.Spectrum(x=spec, y=flux, yerr=fluxerr, xunit='AA', yunit='mag', z=redshift)

AggregateSed - A collection of Segments (Spectra and/or Seds), usually
from one astronomical source. Users may inspect and manipulate each
Segment in the AggregateSed independently.

    >>> # create an AggregateSed
    >>> aggregate_sed = sedstacker.sed.AggregateSed([sed, spectrum])
    >>> print aggregate_sed[0]
      x      y   yerr xunit yunit
    ------ ----- ---- ----- -----
    3823.0 20.43 0.08    AA   mag
    4459.7 20.28 0.08    AA   mag
    5483.8 20.18 0.08    AA   mag
    4779.6 20.18 0.08    AA   mag
    6295.1 22.73 0.08    AA   mag
    7640.8 21.01 0.08    AA   mag
    
    >>> print aggregate_sed[1]
      x      y   yerr
    ------ ----- ----
    3823.0 20.43 0.08
    4459.7 20.28 0.08
    5483.8 20.18 0.08
    4779.6 20.18 0.08
    6295.1 22.73 0.08
    7640.8 21.01 0.08

Stack - A collection of Segments and/or AggregateSeds, usually from different
 observations of one astronomical source, or from individual astronomical sources.
 Stacks may be bulk redshifted and/or normalized together, then statistically combined into
and averaged SED or spectrum.

    >>> # create a Stack
    >>> my_stack = sedstacker.sed.Stack([sed, spectrum])

A Stack is simply a list of SEDs and Spectra. Information
about the individual SEDs can be accessed as one would access data to
a Python list:

    >>> print seds[0].z
    0.668
    >>> print low_z_seds[2].x
    [3823.0, 4459.7, 5483.8, 4779.6]

Then, to combine the SEDs in the Stack, use `sedstacker.sed.stack()`:

    >>> # average the SEDs in my_stack on a linear scale, using a binsize of 2000.0
    >>> stacked_seds = sedstacker.sed.stack(my_stack, 1500.0, 'avg', logbin=False)
    >>> print stacked_seds
      x      y     yerr xunit yunit
    ------ ------ ----- ----- -----
    3823.0 20.355 0.075    AA   mag
    5323.0  20.18   0.0    AA   mag
    6823.0  22.73   0.0    AA   mag
    8323.0  21.01   0.0    AA   mag

### File I/O

Users can load two types of datasets from file: spectra and photometry catalogs.
Both file types are ASCII tables, meaning they're composed of white-spaced
separated columns, with each column having the same number of rows.

A *spectrum file* is a file with at least  two white-space-separated columns:
one with the spectral coordinates (in wavelength), the other with the corresponding
flux coordinates. The file may also have a third column with the corresponding
flux errors. Ex:

    #x            y            yerr
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
        x          y           yerr
    --------- ------------ ------------
    2266.1474  0.032388065 0.0001015389
    2266.9024 0.0082257072 6.264234e-05
    2267.6575  0.026719441 0.0001910421
    2268.4125  0.029819652 0.0007046351
    2269.1676  0.092393999 0.0001372951
    2269.9226 0.0045967875 2.348762e-05
    2270.6777  0.010004328 0.0005329874
    >>>
    >>> #print the spectral values
    >>> print spectrum.x
    [ 2266.1474  2266.9024  2267.6575  2268.4125  2269.1676  2269.9226
      2270.6777]

Users can load spectrum files as SEDs or spectra by assigning the keyword argument *sed_type*.
To store data as a SED object, use "sed":

    >>> sed1 = io.load_sed('data/sed1.txt', sed_type='sed')

To store data as a spectrum, use "spectrum"

    >>> spec1 = io.load_sed('data/spec1.txt', sed_type='spectrum')

Users can assign a redshift to a Sed or Spectrum loaded from file as follows:

    >>> # assign a redshift to a spectrum or SED
    >>> spectrum.z = 0.12

A *photometry catalog* is a file with whitespace-separated columns, each
with the same number of rows, where each row represents a unique astronomical
object. The file must have at least two columns: one with the ID for each object,
and one with fluxes for a specific spectral value. There can be any number of
columns in a Photometry Catalog, but columns must have unique names. The last
line in the commented header is used for the column names. Ex:

    # Photometry data from Bongiorno et al. (2012; DOI: 10.1111/j.1365-2966.2012.22089.x)
    # AGN in COSMOS field
    # ID   z   ucfht errU Bsub  errB Vsub  errV gsub  errg
    2051 0.668 20.43 0.09 20.28 0.09 20.18 0.09 20.18 0.09
    41   0.962 22.73 0.09 22.38 0.09 21.93 0.09 22.44 0.09
    164  0.529 21.16 0.09 21.05 0.09 20.84 0.09 21.01 0.09

To load data from a photometry catalog, a dictionary that maps the
column names to the spectral value, spectral and flux units, and
flux uncertainties must be provided in the following format:

    >>> column_map = {"ucfht":(3823.0,"AA","mag","errU"),
                      "Bsub":(4459.7,"AA","mag","errB"),
                      "Vsub":(5483.8,"AA","mag","errV"),
                      "gsub":(4779.6,"AA","mag","errg"),
                      }

Then, load the catalog, which stores the rows in the photometry catalog
as Sed's in a Stack object:

    >>> seds = io.load_cat('data/seds.cat',
                           column_map=column_map)

Each row is stored as a SED object; the collection of SEDs are then stored
as a Stack object (i.e. `seds` is a
Stack). The other column names in the file are stored as
attributes for each SED stored in the Stack.

If the catalog contains the redshifts in a column named "*z*", the redshifts will be
assigned to the SEDs:

    >>> print seds.z
    [0.668, 0.962, 0.529]

### Example Use Case

The user has a photometry catalog file of SEDs which she wants to
categorize by redshift before stacking.

    >>> from sedstacker.io import load_cat
    >>> from sedstacker.sed import Stack, stack
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
    >>> # categorize seds by their redshifts: sources with z < 1.0,
    >>> # and sources with z > 1.0
    >>> lowz = Stack([sed for sed in seds if sed.z < 1.0])
    >>> highz = Stack([sed for sed in seds if sed.z >= 1.0])
    >>>
    >>> # shift the SEDs to restframe
    >>> lowz_rf = lowz.shift(0.0)
    >>> highz_rf = highz.shift(0.0)
    >>>
    >>> # normalize the SEDs so that the integrated flux of each SED is
    >>> # 1
    >>> norm_lowz_rf = lowz_rf.normalize_by_int()
    >>> norm_highz_rf = highz_rf.normalize_by_int()
    >>>
    >>> # combine the lowz SEDs using weighted average and logarithmic
    >>> # binning
    >>> lowz_stack = stack(norm_lowz_rf, 0.2, 'wavg', logbin=True)


