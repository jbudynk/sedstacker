import logging
import numpy
import types
import os

from astropy.io import ascii
from astropy.io.ascii.core import InconsistentTableError
from astropy.table import Table

from sedstacker.sed import (
    PhotometricPoint, Sed, Spectrum, AggregateSed, create_from_points
    )
from sedstacker.exceptions import (
    NonSupportedFileFormatError,
    NonStandardColumnNamesError,
    PreExistingFileError
    )
from sedstacker.config import NONE_VALS, NUMERIC_TYPES


# Logging handler
_logger = logging.getLogger(__name__)
_formatter = logging.Formatter('%(levelname)s:%(message)s')
_hndlr = logging.StreamHandler()
_hndlr.setFormatter(_formatter)
_logger.addHandler(_hndlr)

# Default units to load data in
_YUNIT = 'erg/s/cm**2/AA'
_XUNIT = 'AA'

def load_sed(filename, xunit=_XUNIT, yunit=_YUNIT, sed_type='spectrum', fmt='ascii'):
    '''
    Reads a file containing SED data adhering to the SED File format and
    returns either a Sed or Spectrum object, depending on the argument
    of 'type'.

    The file must follow the SED File format: a file with at least two
    columns of equal length separated by whitespace representing the spectral
    and flux axes; the first line of the header contains the column names,
    where the spectral, flux and flux-error (if it exists) columns must be
    named 'x', 'y' and 'y_err', respectively.

    Parameters
    ----------
    filename : str
        The filename of the data to read in.
    xunit : str
        The spectral coordinate units. Default is Angstroms.
    yunit : str
        The flux units. Default is erg/s/cm**2/Angstrom.
    sed_type : 'spectrum' or 'sed'
        The argument for sed_type determines whether the object is loaded as a
        Spectrum or Sed object. Default is 'spectrum'.
    fmt : 'ascii' or str
        The file format.
        
        *For release 1.0, only ASCII files will be supported, so* fmt **must**
        *be* 'ascii'.

    Returns
    -------
    sedstacker.sed.Spectrum or sedstacker.sed.Sed
        A Spectrum (if ``sed_type='spectrum'``) or Sed (if ``sed_type='sed'``).

    Examples
    --------
    Load a spectrum from file.

    >>> # The ASCII file follows the SED File Format
    >>>
    >>> more data/3c273.huit.dat
    # x y y_err
    815.0576782227 -1.2162471550725e-14 7.6739829768524e-16
    815.5765380859 -1.2152171630087e-14 7.6674777638175e-16
    816.0953369141 -1.2141980129665e-14 7.6609725507826e-16
    816.6141967773 -1.2131680209027e-14 7.6544673377477e-16
    817.1329956055 -1.2121488708605e-14 7.6479621247127e-16
    817.6518554688 3.9157045661486e-15 8.3851111817856e-15
    ...
    >>> 
    >>> from sedstacker.io import load_sed
    >>> spec_3c273 = load_sed('data/3c273.huit.dat')
    >>> 
    >>> print spec_3c273
          x               y                 yerr      
    ------------- ------------------ -----------------
    815.057678223 -1.21624715507e-14 7.67398297685e-16
    815.576538086 -1.21521716301e-14 7.66747776382e-16
    816.095336914 -1.21419801297e-14 7.66097255078e-16
    816.614196777 -1.21316802090e-14 7.65446733775e-16
    ...                ...               ...
    1874.50854492 -9.11326136083e-15 5.74952412069e-16
    1875.02734375 -9.16270097989e-15 5.78096598369e-16
    1875.54614258 -9.21571846613e-15 5.81457625104e-16
    1876.06506348  -9.2727475004e-15 5.85035492273e-16
    >>> 
    >>> # View the flux data
    >>> 
    >>> print spec_3c273.y
    [ -1.21828546e-14  -1.21726631e-14  -1.21624716e-14 ...,  -9.16270098e-15
      -9.21571847e-15  -9.27274750e-15]

    Notes
    -----
    - If the file is the result of a stacked SED, then the 'counts' column will
    also be stored in the Sed or Spectrum object as attribute *count*.

    '''

    # Check that the data file format is an acceptable file format
    if not _check_file_format(fmt):
        raise NonSupportedFileFormatError(fmt)

    # Read data from file into a dictionary with three key-value pairs:
    # {'x':spec, 'y':flux, 'y_err':flux-err}
    if fmt == 'ascii':
        sed_data = _read_ascii(filename) #, aux_data=aux_data)

    # Storing the SED data in a Sed or Spectrum object
    if sed_type == 'spectrum':
        spectrum = Spectrum()
        spectrum.x = sed_data['x']
        spectrum.y = sed_data['y']
        spectrum.yerr = sed_data['y_err']
        if sed_data['counts'] is not None:
            setattr(spectrum, 'counts', sed_data['counts'])
        spectrum.xunit = xunit
        spectrum.yunit = yunit

        logger.info(' Created Spectrum object.')
        return spectrum

    elif sed_type == 'sed':
        points = []
        for i in range(len(sed_data['x'])):
            points.append(PhotometricPoint(x=sed_data['x'][i],
                                           y=sed_data['y'][i],
                                           yerr=sed_data['y_err'][i],
                                           xunit=xunit,
                                           yunit=yunit))
        sed = create_from_points(points)
        if sed_data['counts'] is not None:
            setattr(sed,'counts',sed_data['counts'])
        logger.info(' Created Sed object.')
        return sed

    else:
        raise ValueError('Invalid argument for keyword sed_type.'+ 
                         'Options: "spectrum" or "sed".')
	

def load_dir(directory, xunit=_XUNIT, yunit=_YUNIT, sed_type='spectrum', fmt='ascii'):

    '''Loads a directory of spectra into one AggregateSed.

    The files must follow the SED File format: files with at least two columns
    of equal length separated by whitespace representing the spectral and flux
    axes; the first line of the header contains the column names, where the
    spectral, flux and flux-error (if it exists) columns must be named 'x', 'y'
    and 'y_err', respectively.

    Parameters
    ----------
    directory : str
        The path to the directory of the data.
    xunit : str
        The spectral coordinate units. Default is Angstroms.
    yunit : str
        The flux units. Default is erg/s/cm^2/Angstrom.
    sed_type : 'spectrum' or 'sed'
        The argument for sed_type determines whether the object is loaded as a
        Spectrum or Sed object. Default is 'spectrum'.
    fmt : 'ascii' or str
        The file format.
        
        *For release 1.0, only ASCII files will be supported, so* fmt **must**
        *be* 'ascii'.

    Returns
    -------
    list of sedstacker.sed.Spectrum or sedstacker.sed.Sed
        A list of Spectra or Sed objects, depending on the value of parameter
        *sed_type*.

    Notes
    -----
    - All files must be in the same units, and of the same sed_type (i.e. you
    cannot mix Sed and Spectrum objects together via this function).
    - If the file is the result of a stacked SED, then the 'counts' column
    will also be stored in the Sed or Spectrum object as attribute *count*.

    Examples
    --------
    Say directory 'my/data/directory/' contains data files of spectra we'd like
    to load for analysis with SEDstacker. The files are in the same units, with
    the spectral axis in Angstroms and the flux axis in erg/s/cm**2/Angstrom.
    Some files have flux-error columns, some do not. Here's an example of one
    of the files in the directory::

         x y y_err
         2266.1474     0.032388065  1.015389e-4
         2266.9024    0.0082257072  6.264234e-5
         2267.6575     0.026719441  1.910421e-4
         2268.4125     0.029819652  7.046351e-4
         2269.1676     0.092393999  1.372951e-4
         2269.9226    0.0045967875  2.348762e-5
         2270.6777     0.010004328  5.329874e-4
         ...

    In Python:
    
    >>> from sedstacker.io import load_dir
    >>> spectra = load_dir('my/data/directory/', sed_type='spectrum')
    
    'spectra' is a list of Spectrum objects representing the data from each
    file in 'my/data/directory/'.

    '''
    direc = os.listdir(directory)

    spectra = []
    for files in direc:
        spec.append(load_sed(directory+files, 
                             xunit=xunit, yunit=yunit, sed_type=sed_type, 
                             fmt=fmt)
                    )

    return AggregateSed(spectra)


def load_cat(filename, column_map, fmt='ascii', **kwargs):

    '''
    Reads a photometry catalog and returns an AggregateSed (if the file
    contains multiple SEDs) or a Sed (if file contains just one SED).

    Parameters
    ----------
    filename : str
        The name of the input file.
    column_map : dict
        A dictionary that assigns the spectral values, units and flux units
        for each photometric band in the catalog. The key must be the name of
        the column, and the value must be a four-element tuple in the order
        spectral value, spectral unit, flux unit, and flux error column name.
        
        Ex: Catalog with two photometric bands, named 'sdss_g' and 'mips24'.
        The file looks like
        
        .. code-block::
            # ID z sdss_g sdss_gerr mips24 mips24err
            1 0.12 
        
        >>> column_map = {'sdss_g':(4770., 'AA', 'mag', 'sdss_gerr'),
                          'mips24':(240000., 'AA', 'mag','mips24err')
                          } 

        If there is no error column associated with a flux column, the fourth
        element in the tuple (the error column name) should be "None". For Ex,
        if sdss_g had no error column, we'd write:

        >>> column_map = {'sdss_g':(4770., 'AA', 'mag', None),
                          'mips24':(240000., 'AA', 'mag','mips24err')
                          }     
    fmt : str
        The file format of the input file.
        
        *For release 1.0, only ASCII files will be supported, so* fmt **must**
        *be* 'ascii'.

    Returns
    -------
    sedstacker.sed.AggregateSed  or  sedstacker.sed.Sed
        

    Examples
    --------
    Say we have photometric data for 6 sources, ranging from the UV to near-IR
    with the following format:
    
    >>> more photometry-catalog.dat
    #ID    z   ucfht errU  Bsub errB  Vsub errV  gsub errg  rsub errR  isub errI  zsub errz
    #lambda_eff 3823.0     4459.7     5483.8     4779.6     6295.1     7640.8     9036.9
    2051 0.668 20.43 0.09 20.28 0.09 20.18 0.09 20.18 0.09 20.26 0.09 20.13 0.09 20.01 0.09
      41 0.962 22.73 0.09 22.38 0.09 21.93 0.09 22.44 0.09 21.77 0.09  null null  21.4 0.09
     164 0.529 21.16 0.09 21.05 0.09 20.84 null 21.01 0.09 20.51 0.09 20.18 0.09 19.96 0.09
     106   0.9 27.72 0.63 26.82 0.36 25.31 0.16 26.68 0.44 24.42 0.09 23.09 0.09  22.3 0.09
     194 1.456 24.78 0.11 24.54 0.09 24.62 0.13 24.65 0.13 24.12 0.09 23.70 0.09 22.99 0.09
     331 2.415 26.89 0.68 25.92  0.3 25.55 0.30 25.75 0.35 25.19 0.22 24.97 0.25 25.21 0.63

    The first commented row contains the column names; the second row has the
    effective wavelengths of the photometric bands. Column names must follow
    Python's identifier syntax [0]_.
    
    Missing or unknown data should be represented by either
    'null', 'None', 'none', '-99', '-99.', 'nan' or 'NaN' in the data file.
    If the null value cell is a flux value, then that photometric point will be
    ignored from the Sed. Otherwise, the value will be converted to numpy.nan.
    For example, in the file above, SED ID 41 will not have a photometric point
    at 'isub' (7640.8 AA). ID 164's 'Vsub' flux-error value will be numpy.nan.

    Next, we create the column map to assign the spectral values, units, and
    flux units to each photometric band.

    >>> column_map = {"ucfht":(3823.0,"AA","mag","errU"),
                      "Bsub":(4459.7,"AA","mag","errB"),
                      "Vsub":(5483.8,"AA","mag","errV"),
                      "gsub":(4779.6,"AA","mag","errg"),
                      "rsub":(6295.1,"AA","mag","errR"),
                      "isub":(7640.8,"AA","mag","errI"),
                      "zsub":(9036.9,"AA","mag","errz")
                      }

    Load the data into some variable:
    
    >>> from sedstacker.io import load_cat
    >>> seds = load_cat('photometry-catalog.dat', column_map)

    In this case, seds is an AggregateSed, which is a list of Sed objects. The
    AggregateSed's spectral, flux and flux-errors are lists of the individual
    SEDs' spectral, flux, and flux-errors:

    >>> # View the spectral axes
    >>> 
    >>> seds.x
    [array([ 3823. ,  4459.7,  5483.8,  4779.6,  6295.1,  7640.8,  9036.9]),
    array([ 3823. ,  4459.7,  5483.8,  4779.6,  6295.1,  9036.9]),
    array([ 3823. ,  4459.7,  5483.8,  4779.6,  6295.1,  7640.8,  9036.9]),
    array([ 3823. ,  4459.7,  5483.8,  4779.6,  6295.1,  7640.8,  9036.9]),
    array([ 3823. ,  4459.7,  5483.8,  4779.6,  6295.1,  7640.8,  9036.9]),
    array([ 3823. ,  4459.7,  5483.8,  4779.6,  6295.1,  7640.8,  9036.9])]
    >>> 
    >>> # View the flux array for the third SED (ID 164).
    >>> 
    >>> seds[2].y
    array([ 21.16,  21.05,  20.84,  21.01,  20.51,  20.18,  19.96])
    
    All columns are added as attributes to the individual Seds in the resultant
    AggregateSed. The column names are converted to lowercase before being
    assigned as an attribute.

    >>> seds[1].id
    41
    >>> seds[4].z
    1.456

    Notes
    -----
    - If a column of redshift values exists in the photometry catalog, label
    the column as 'z' for load_cat() to read the redshift information in correctly.

    References
    ----------
    ..[0] http://docs.python.org/2/reference/lexical_analysis.html#identifiers

    '''

    if not _check_file_format(fmt):
        raise NonSupportedFileFormatError('%s is a non-supported file format.' % fmt)

    try:
        catalog = ascii.read(filename,
                             format='commented_header',
                             header_start = -1)

    # catch unequal lengths of table columns or rows
    except InconsistentTableError:
        try:
            catalog = ascii.read(filename)
        except InconsistentTableError:
            raise InconsistentTableError('File does not follow Photometry '+
                                      'Catalog format. Check that all '+
                                      'columns have the same number of rows.')
    # catch bad column names
    except KeyError, e:
        if e.message == 0:
            raise NonStandardColumnNamesError
        else:
            raise

    column_names = catalog.colnames

    # catch bad column names
    if column_names[0] == 'col1':
        catalog = ascii.read(filename, format='commented_header')
        column_names = catalog.colnames
    if column_names[0] == 'col1':
        raise ValueError('Column names in file do not match column names in '+
                         'column_map.')
    if _check_float_colnames(column_names):
        raise NonStandardColumnNamesError
        
    seds = []

    for row in range(len(catalog)):

        x = []
        y = []
        yerr = []
        yerr_name = []
        xunit = []
        yunit = []
        z = []

        sed = Sed()

        for name in column_names:

            if name in column_map:

                x.append(float(column_map.get(name)[0]))
                xunit.append(column_map.get(name)[1])
                yunit.append(column_map.get(name)[2])

                y.append(catalog[name].data[row])

                # deals with case if no flux error is associated to a flux
                if column_map.get(name)[3] is None:
                    yerr.append(numpy.nan)
                else:
                    yerr_name.append(column_map.get(name)[3])
                    yerr.append(catalog[yerr_name[len(yerr_name)-1]].data[row])
            
            setattr(sed, name.lower(), catalog[name].data[row])

        for i in range(len(x)):
            # for flux and fluxerrs with null values,
            # set the value to numpy.nan
            if y[i] in NONE_VALS:
                continue
            else:
                y[i] = numpy.float_(y[i])
            if yerr[i] in NONE_VALS:
                 yerr[i] = numpy.nan
            else:
                yerr[i] = numpy.float_(yerr[i])
            sed.add_point(PhotometricPoint(x=x[i],y=y[i],yerr=yerr[i],
                                           xunit=xunit[i],yunit=yunit[i]))

        seds.append(sed)

    if len(seds) > 1:
        return AggregateSed(seds)
    if len(seds) == 1:
        return seds[0]


def _check_file_format(fmt):
    '''Check that fmt is in the list of acceptable file formats.'''
    FORMATS = ('ascii',)
    return fmt in FORMATS


def _read_standard_colnames(sed_file):
    spec = numpy.array(sed_file['x'].data)
    flux = numpy.array(sed_file['y'].data)
    if 'y_err' in sed_file.colnames:
        fluxerr = numpy.array(sed_file['y_err'].data)
    else:
        _no_fluxerror_column_info()
        fluxerr = None
    if 'counts' in sed_file.colnames:
        counts = numpy.array(sed_file['counts'].data)
    else:
        counts = None

    return dict(x=spec, y=flux, y_err=fluxerr, counts=counts)


def _read_nonstandard_colnames(sed_file):
    cols = sed_file.colnames
    spec = numpy.array(sed_file[cols[0]].data)
    flux = numpy.array(sed_file[cols[1]].data)
    if len(cols) > 2:
        fluxerr = numpy.array(sed_file[cols[2]].data)
    else:
        _no_fluxerror_column_info()
        fluxerr = [None]*spec.size
    counts = None
    
    # if no column names are provided, ascii.read() makes the first
    # row of the file into the column names
    if _no_colnames(cols):
        spec = numpy.insert(spec, 0, float(cols[0]))
        flux = numpy.insert(flux, 0, float(cols[1]))
        try:
            fluxerr = numpy.insert(fluxerr, 0, float(cols[2]))
        except IndexError:
            fluxerr = numpy.insert(fluxerr, 0, None)

    return dict(x=spec, y=flux, y_err=fluxerr, counts=counts)
    

def _read_ascii(filename):
    '''
    Returns a dictionary representation of the SED data from an ASCII file
    following the SED File format.

    Parameters
    ----------
    filename : str
        Name of ASCII file.

    Returns
    -------
    dict('x':spec, 'y':flux, 'y_err':fluxerr)

    Notes
    -----
    This does NOT store any other columns of information (11/26/2013)

    '''

    try:
        sed_file = ascii.read(filename, format='commented_header')
        d = _read_standard_colnames(sed_file)

    except KeyError:
        _non_standard_column_names_warning(filename)
        
        sed_file = ascii.read(filename, format='commented_header')
        d = _read_nonstandard_colnames(sed_file)

    except InconsistentTableError:
        sed_file = ascii.read(filename, format='basic')

        try:
            d = _read_standard_colnames(sed_file)

        except KeyError:
            _non_standard_column_names_warning(filename)
            d = _read_nonstandard_colnames(sed_file)

    # to deal with 'null' values in table
    try:    
        for i, val in enumerate(d['y_err']):
            if val in NONE_VALS:
                d['y_err'][i] = numpy.nan
            else:
                d['y_err'][i] = numpy.float_(d['y_err'][i])
        for i, val in enumerate(d['y']):
            if val in NONE_VALS:
                d['y'][i] = numpy.nan
            else:
                d['y'][i] = numpy.float_(d['y'][i])

    # if there's no y_err column in file
    # set the fluxerr to None
    except TypeError:
        d['y_err'] = numpy.array([None]*d['x'].size)
        print d['y_err']

    return dict(x=d['x'], y=d['y'], y_err=d['y_err'], counts=d['counts'])


def _no_fluxerror_column_info():
    logger.info(' No flux-error column found. ' 
                'Sed/Spectrum object\'s \'yerr\' attribute set to None.'
                )


def _non_standard_column_names_warning(filename):
    logger.warning(' Column names in "%s" do not adhere to SED File format.'
                   'Reading the first column as spectral values, second column '
                   'as fluxes, and third column, if present, as flux-errors.'
                   % os.path.split(filename)[1]
                   )

def _no_colnames(cols):

    count=0
    for name in cols:
        try:
            if float(name):
                count += 1
        except ValueError:
            pass

    if count != len(cols):
        return False
    
    return True


def _check_float_colnames(cols):
    
    count=0
    for name in cols:
        try:
            count += float(name)
        except ValueError:
            pass

    if count > 0:
        return True

    return False
