sedstacker
==========

A Python toolkit for statistically combining multiple spectral energy distributions (SEDs).

Installation
------------

From the command line, run

    git clone --recursive git://github.com/jbudynk/sedstacker.git

`cd` into `sedstacker` and run

    python setup.py sedstacker install

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

sedstacker was developed for interest in creating SED template models,
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



### Example Use Case

The user has a photometry catalog file of SEDs which she wants to
categorize by redshift before stacking. After loading in the data file through other means (astropy.io, pyfits, etc.),
the user creates a Stack object from her data, called `seds`.

    >>> from sedstacker.sed import Stack, stack
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


