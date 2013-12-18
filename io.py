import logging
import numpy
import types
import os.path

from astropy.io import ascii
from astropy.io.ascii.core import InconsistentTableError
from astropy.table import Table

from sedstacker.sed import PhotometricPoint, Sed, Spectrum, AggregateSed, create_from_points
from sedstacker.exceptions import NonSupportedFileFormatError, NonStandardColumnNamesError, PreExistingFileError
from sedstacker.config import NONE_VALS


# (12/16/2013 - took out **kwargs from load_cat(). Haven't tested it, don't know if **kwargs would work.)


def load_sed(filename, xunit='AA', yunit='erg/s/cm**2/AA', sed_type='spectrum', fmt='ascii'): #aux_data=None
    '''Reads a file containing SED data adhering to the SED File format and returns either a Sed or Spectrum object, depending on the argument of 'type'.

    >>> spec_3c273 = load_sed('data/3c273.huit.dat')
    >>> print spec_3c273.y
    [ -1.21828546e-14  -1.21726631e-14  -1.21624716e-14 ...,  -9.16270098e-15
      -9.21571847e-15  -9.27274750e-15]
    
    Args:
        filename (str): the filename of the data to read in.

    Kwargs:
        xunit (str): The spectral coordinate units. Default is Angstroms.
        yunit (str): The flux units. Default is erg/s/cm^2/Angstrom.
        sed_type (str): 'spectrum/sed'. The argument for sed_type determines whether the object is loaded as a Spectrum or Sed object. Default is 'spectrum'.
        fmt (str): 'ascii'. The default file format. For release 1.0, only ASCII files will be supported, so this must remain unchanged.

    The file must follow the SED File format: a file with at least two columns of equal length separated by whitespace representing the spectral and flux axes; the first line of the header contains the column names, where the spectral, flux and flux-error (if it exists) columns must be named 'x', 'y' and 'y_err', respectively. If the file is the result of a stacked SED, then the 'counts' column will also be stored in the Sed or Spectrum object.'''

    # Check that the data file format is an acceptable file format
    if not check_file_format(fmt):
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
#        if type(aux_data) is types.ListType:
#            for data in aux_data:
#                if data is not types.StringType:
#                    raise TypeError('aux_data must be a list of strings of column names from input file.')
#                else:
#                    setattr(spectrum, data, sed_data[data].data)
#        elif aux_data is None:
#            pass
#        else:
#            raise TypeError('aux_data must be a list of strings of column names in input file.')
        logging.info('Created Spectrum object.')
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
        logging.info('Created Sed object.')
        return sed

    else:
        raise ValueError('Invalid argument for keyword sed_type. Options: "spectrum" or "sed".')
	

def load_cat(filename, column_map, fmt='ascii', **kwargs):
    '''Reads a photometry catalog and returns a Sed object (if file contains just one SED) or an AggregateSed object (if the file contains multiple SEDs).

    Args:
        filename (str): The name of the output file.
        column_map (dict): A dictionary that assigns the spectral values, units and flux units for each photometric band in the catalog. The key must be the name of the column, and the value must be a four-element tuple in the order spectral value, spectral unit, flux unit, and flux error column name. Ex: Catalog with two photometric bands, named 'sdss_g' and 'mips24':
        ..code:: column_map = {'sdss_g':(4770., 'AA', 'mag', 'sdss_gerr'), 'mips24':(24., 'micron', 'uJy','mips24err')} 
        If there is no error column associated with a flux column, the fourth element in the tuple (the error column name) should be "None". For Ex, if sdss_g had no error column, we'd write:
        ..code:: column_map = {'sdss_g':(4770., 'AA', 'mag', None), 'mips24':(24., 'micron', 'uJy','mips24err')} 
    Kwargs:
        fmt (str): The file format of the input file. The default file format is 'ascii'. For release 1.0, only ASCII files will be supported, so this must remain unchanged.
        '''
    # **kwargs: keyword arguments accepted by astopy.io.ascii.read() may be used to further specify the information you wish to load.

    if not check_file_format(fmt):
        raise NonSupportedFileFormatError('%s is a non-supported file format.' % fmt)

    try:
        catalog = ascii.read(filename, header_start = -1) #kwargs
    # catch unequal lengths of table columns or rows
    except InconsistentTableError, e:
        try:
            catalog = ascii.read(filename)
        except InconsistentTableError, e:
            print e
            raise InconsistentTableError('File does not follow Photometry Catalog format.'+
                                         ' Check that all columns have the same number of rows.')
    # catch bad column names
    except KeyError, e:
        if e.message == 0:
            raise NonStandardColumnNamesError
        else:
            raise
        

    column_names = catalog.colnames

    # catch bad column names
    if column_names[0] == 'col1':
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
            if yerr[i] in NONE_VALS:
                 yerr[i] = numpy.nan
            else:
                yerr[i] = numpy.float_(yerr[i])
            if y[i] in NONE_VALS:
                y[i] = numpy.nan
            else:
                y[i] = numpy.float_(y[i])
            sed.add_point(PhotometricPoint(x=x[i],y=y[i],yerr=yerr[i],xunit=xunit[i],yunit=yunit[i]))

        seds.append(sed)

    if len(seds) > 1:
        return AggregateSed(seds)
    if len(seds) == 1:
        return seds[0]


def check_file_format(fmt):
    '''Check that fmt is in the list of acceptable file formats.'''
    FORMATS = ('ascii',)
    return fmt in FORMATS


def _read_ascii(filename):
    '''Returns a dictionary representation of the SED data from an ASCII file following the SED File format.
    Args:
        filename (str): Name of ASCII file.
    Returns:
        dict('x':spec, 'y':flux, 'y_err':fluxerr)

        Note:
            This does NOT store any other columns of information (11/26/2013)'''

    try:
        sed_file = ascii.read(filename) #header_start = -1)

        spec = numpy.array(sed_file['x'].data)
        flux = numpy.array(sed_file['y'].data)
        if 'y_err' in sed_file.colnames:
            fluxerr = numpy.array(sed_file['y_err'].data)
        else:
            no_fluxerror_column_warning()
            fluxerr = None
        if 'counts' in sed_file.colnames:
            counts = numpy.array(sed_file['counts'].data)
        else:
            counts = None

    except KeyError:
        if 'x' not in sed_file.colnames:
            non_standard_column_names_warning(filename)

            sed_file = ascii.read(filename) #header_start = -1)

            cols = sed_file.colnames
            sed_file.rename_column(cols[0],'x')
            sed_file.rename_column(cols[1],'y')
            if len(cols) > 2:
                sed_file.rename_column(cols[2],'y_err')

            spec = numpy.array(sed_file['x'].data)
            flux = numpy.array(sed_file['y'].data)
            if 'y_err' in sed_file.colnames:
                fluxerr = numpy.array(sed_file['y_err'].data)
            else:
                no_fluxerror_column_info()
                fluxerr = None
            counts = None

    except InconsistentTableError, e:
        print e
        raise

    # to deal with 'null' values in table
    try:    
        for i, val in enumerate(fluxerr):
            if val in NONE_VALS:
                fluxerr[i] = numpy.nan
            else:
                fluxerr[i] = numpy.float_(fluxerr[i])
        for i, val in enumerate(flux):
            if val in NONE_VALS:
                flux[i] = numpy.nan
            else:
                flux[i] = numpy.float_(flux[i])

    # if there's no y_err column in file
    # set the fluxerr to None
    except TypeError, e:
        fluxerr = None

    return dict(x=spec, y=flux, y_err=fluxerr, counts=counts)


def no_fluxerror_column_info():
    logging.info(' No flux-error column found. ' 
                 'Sed/Spectrum object\'s \'yerr\' attribute set to None.'
                 )


def non_standard_column_names_warning(filename):
    logging.warning(' Column names in "%s" do not adhere to SED File format.\n'
                    'Reading the first column as spectral values, second column '
                    'as fluxes, and third column, if present, as flux-errors.'
                    % filename
                    )


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
