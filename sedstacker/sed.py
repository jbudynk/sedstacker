import logging
import numpy
import os.path
import types
from math import log10
from bisect import bisect_left, bisect_right

from astropy.io import ascii
from astropy.table import Table

from sedstacker import calc
from sedstacker.config import NUMERIC_TYPES
from sedstacker.exceptions import NoRedshiftError, InvalidRedshiftError, SegmentError, OutsideRangeError, NotASegmentError, PreExistingFileError, \
    BadRangesError
import time


logger=logging.getLogger(__name__)
formatter=logging.Formatter('%(levelname)s:%(message)s')
hndlr=logging.StreamHandler()
hndlr.setFormatter(formatter)
logger.addHandler(hndlr)


class PhotometricPoint(object):
    '''Represents a photometric point on a SED.

    To create a PhotometricPoint

        >>> point = PhotometricPoint(x=3823.0, y=1e-16, yerr=1e-18)
        >>> print point
        (3823 erg/s/cm**2/AA, 1e-16 +/- 1e-18 erg/s/cm**2/AA)
        >>> print point.x
        3823.0

     Parameters
     ----------
     x : float or int
         The spectral coordinate
     y : float or int
         The flux coordinate
     yerr : float, int or None.
         The corresponding error to *y*.
     xunit : string or None
         The unit of the spectral coordinate
     yunit : string or None
         The unit of the flux coordinate

     Attributes
     ----------
     mask : bool
         If True, mask the point. Default is False.

    '''

    def __init__(self, x=None, y=None, yerr=None, xunit='AA', yunit='erg/s/cm**2/AA'):
                
        self.x = x
        self.y = y
        if yerr is None:
            self.yerr = numpy.nan
        else:
            self.yerr = yerr
        self.xunit = xunit
        self.yunit = yunit
        self.mask = False


    def __str__(self):

        if self.yerr != None:
            return '(%g %s, %g +/- %g %s)' (self.x, self.xunit, self.y, self.yerr, self.yunit)
        else:
            return '(%g %s, %g %s)' % (self.x, self.xunit, self.y, self.yunit)


class Segment(object):

    def shift(self, z0, correct_flux=True):
        raise NotImplementedError('shift() is implemented only for instantiated Segment subclass objects.')

    def normalize_at_point(self, x0, y0, norm_operator=0, correct_flux=False):
        raise NotImplementedError('normalize_at_point() is implemented only for instantiated Segment subclass objects.')

    def normalize_by_int(self, minWavelength='min', maxWavelength='max', correct_flux=False, z0=None):
        raise NotImplementedError('normalize_by_int() is implemented only for instantiated Segment subclass objects.')

#    def correct_flux(self):
#        raise NotImplementedError('corrext_flux() is implemented only for instantiated Segment subclass objects.')

#    def get_unit(self, attr):
#        raise NotImplementedError('get_unit() is implemented only for instantiated Segment subclass objects.')

#    def set_unit(self, attr, unit=0):
#        raise NotImplementedError('set_unit() is implemented only for instantiated Segment subclass objects.')


class Spectrum(Segment):
    '''A Spectrum is a spectrum in the astrophysical sense, meant to represent data taken from a spectrometer.

    Parameters
    ----------
    x : array_like
        The spectral coordinates. Default value is an empty list, [].
    y : array_like
        The flux values. Default value is an empty list, [].
    yerr : array_like of float or int; or None
        The errors on the flux values. Default value is None.
    xunit : array_like of str
        The spectral coordinate units. Default value is ['AA'].
    yunit : array_like of str
        The flux coordinates. Default value is ['erg/s/cm**2/AA'].
    z : float, int
        The redshift of the Sed. Default value is None.

    Examples
    --------
    Create a Spectrum object::

        # Create dummy spectral data
        import numpy
        wavelength = numpy.arange(1200, 10000, 1)
        flux = numpy.random.rand(wavelength.size)
        flux_err = flux*0.01
        
        # Create a Spectrum of an object at redshift 0.32
        from sedstacker.sed import Spectrum
        spectrum = Spectrum(x=wavelength, y=flux, yerr=flux_err, xunit="Angstrom", yunit="None", z=0.32)
        
    View the Spectrum's data:

        >>> # See the spectral, flux and flux-errors in tabular format
        >>> print spectrum
         x          y               yerr      
        ---- --------------- -----------------
        1200   0.17363701609   0.0017363701609
        1201  0.829659253106  0.00829659253106
        1202  0.721316944384  0.00721316944384
        1203  0.340523430975  0.00340523430975
        ...             ...               ...
        9997  0.796084449833  0.00796084449833
        9998  0.162571575683  0.00162571575683
        9999 0.0378459454457 0.000378459454457
        >>> 
        >>> # Access the spectral axis
        >>> spectrum.x
        array([1200, 1201, 1202, ..., 9997, 9998, 9999])
        >>> 
        >>> # Check the redshift
        >>> spectrum.z
        0.32

    '''

    def __init__(self, x=[], y=[], yerr=None, xunit='AA', yunit='erg/s/cm**2/AA', z=None):

        if len(x) != len(y):
            raise SegmentError('x and y must be of the same length.')

        self.x = numpy.array(x)
        self.y = numpy.array(y)

        if yerr is None:
            self.yerr = numpy.array([numpy.nan]*len(y))
        elif type(yerr) in NUMERIC_TYPES:
            self.yerr = numpy.array([float(yerr)]*len(y))
        elif len(yerr) == len(y):
            self.yerr = numpy.array(yerr)
        else:
            raise SegmentError('y and yerr must be of the same length.')

        self.xunit = xunit
        self.yunit = yunit
        self.z = z
        self.counts = None
        self.norm_constant = 1.0


    @property
    def z(self):
        return self._z
    @z.setter
    def z(self, val):
        if isinstance(val, types.NoneType):
            self._z = val
        elif type(val) not in NUMERIC_TYPES:
            raise InvalidRedshiftError(0)
        elif val < 0:
            raise InvalidRedshiftError(1)
        else:
            self._z = numpy.float_(val)
    @z.deleter
    def z(self):
        logging.info('Setting z to None.')
        self._z = None


    def __str__(self):
        # Prints out the spectral, flux, and flux-error data as an AstroPy Table
        data = Table([self.x, self.y, self.yerr], names=('x','y','yerr'), meta={'z':self.z})
        return data.__str__()


    def shift(self, z0, correct_flux=True):
        '''Redshifts the spectrum by means of cosmological expansion.

        Parameters
        ----------
        z0 : float, int
            Target redshift to shift the SED/spectrum to
        correct_flux : bool
            If True, the flux will be corrected for the intrinsic dimming/brightening due to shifting the spectrum. If False, only the spectral coordinates will be shifted; the flux remains the same.

        Returns
        -------
        shifted_spectrum : sedstacker.sed.Spectrum
            A new Spectrum object with the redshifted spectrum.

        Raises
        ------
        sedstacker.exceptions.InvalidRedshiftError
            If attribute *z* or parameter *z0* is of non-numeric type or is negative.
        sedstacker.exceptions.NoRedshiftError
            If attribute *z* is *None*

        Examples
        --------
        >>> # Shift the spectrum to rest frame
        >>> 
        >>> rf_spectrum = spectrum.shift(0)
        >>> 
        >>> # Shift the spectrum to rest frame, without correcting the flux
        >>> 
        >>> rf_spectrum = spectrum.shift(0, correct_flux=False)

        '''

        if isinstance(self.z, types.NoneType):
            raise NoRedshiftError
        if (not type(z0) in NUMERIC_TYPES) or (not type(self.z) in NUMERIC_TYPES):
            raise InvalidRedshiftError(0)
        if z0 < 0.0 or self.z < 0.0:
            raise InvalidRedshiftError(1)

        if correct_flux:
            spec_z0, flux_z0 = shift(self.x, self.y, self.z, z0)
        else:
            spec_z0 = (1 + z0) * self.x / (1+self.z)
            flux_z0 = self.y

        spec = Spectrum(x=spec_z0, y=flux_z0, yerr=self.yerr,
                        xunit=self.xunit, yunit=self.yunit, z=z0)
        # keep attributes of old sed
        _get_setattr(spec,self)

        return spec


    def normalize_at_point(self, x0, y0, dx=50, norm_operator=0, correct_flux=False, z0=None):
        '''Normalizes the spectrum such that at spectral coordinate x0,
        the flux of the spectrum is y0.

        Notes
        -----
        normalize_at_point() takes the average flux within a range of
        spectral values [x0-dx, x0+dx] centered on x0 as the observed
        flux at x0.

        Parameters
        ----------
        x0 : float, int
            The spectral coordinate to normalize the SED at. x0 is in
            Angstroms.
        y0 : float, int
            The flux value to normalize the SED to.
        dx : float, int
            The number of spectral points to the left and right of x0,
            over which the average flux is measured. If no points fall 
            within the range [x0-dx,x=+dx], then OutsideRangeError is 
            raised, and the normalization is aborted.
        norm_operator : int
            operator used for scaling the spectrum to y0.
                - 0 = *[default]* multiply the flux by the normalization 
                constant
                - 1 = add the normalization constant to the flux
        correct_flux : bool
            To correct for flux dimming/brightening due to redshift. 
            Meant for SEDs that were shifted only by wavelength (i.e. 
            the flux was not corrected for the intrinsic dimming/brightening 
            due to redshift). If ``correct_flux = True``, then the flux 
            is corrected so that the integrated flux at the current redshift 
            is equal to that at the original redshift. Default value is False.
        z0 : float, int
            The original redshift of the source. Used only if 
            ``correct_flux = True``.

        Returns
        -------
        norm_spectrum : sedstacker.sed.Spectrum
            A new Spectrum object of the normalized old Spectrum with 
            attribute `norm_constant`.

        Raises
        ------
        sedstacker.exceptions.SegmentError
            If the Spectrum has less than 4 points

        Examples
        --------
        >>> # initializing dummy spectrum
        >>> 
        >>> x = numpy.arange(3000, 9500, 0.5)
        >>> y = numpy.linspace(1, 1000, x.size)
        >>> yerr = y*0.01
        >>> spec = Spectrum(x=x,y=y,yerr=yerr,z=0.3)
        >>> 
        >>> # normalize the spectrum "spec" to 1.0 at 3600.0
        >>> 
        >>> norm_spec = spec.normalize_at_point(3600.0, 1.0, norm_operator=0, dx=70)
        >>> norm_spec.y
        >>> array([  0.01072261,   0.01154666,   0.01237072, ...,  10.72095858,
                     10.72178263,  10.72260668])
        >>> 
        >>> # Check the normalization constant
        >>> 
        >>> norm_spec.norm_constant
        0.010722606684739769

        '''

        numpy.seterr(invalid='raise')

        if len(self.x) < 4:
            raise SegmentError('Spectrum object must have 4 or more points to use normalize_at_point().')

        y0 = numpy.float_(y0)
        x0 = numpy.float_(x0)
        dx = numpy.float_(dx)
        flux = numpy.ma.masked_invalid(self.y)
        fluxerr = self.yerr
#        spec = self.x

#        x0_idx = find_nearest(spec, x0)
#        idx = (x0_idx-dx, x0_idx+dx)
#        
#        if idx[0] < 0 and idx[1] >= spec.size:
#            raise OutsideRangeError
#        elif idx[0] < 0:
#            idx[0] = min(spec)
#        elif idx[1] >= spec.size:
#            idx[1] = max(spec)
        # to print out ranges used in the normalization.
        # Has no effect on the ranges used.
#        if (idx[0] < 0) or (idx[1] >= spec.size):
#            high_lim = min([idx+dx], max(spec))
#            low_lim = max([idx-dx], min(spec))
#            logger.warning(' Spectrum does not cover full range used for '+
#                           'determining normalization constant. Spectral '+
#                           'range used: [{low}:{high}]'.format(low=repr(low_lim), high=repr(high_lim)))

        spec_indices = find_range(self.x, x0-dx, x0+dx+1)
        if spec_indices == (-1, -1):
            raise OutsideRangeError
        elif spec_indices[0] == -1:
            spec_indices[0] = min(self.x)
        elif spec_indices[1] == -1:
            spec_indices[1] = max(self.x)
        if (x0-dx < min(self.x)) or (x0+dx > max(self.x)):
            high_lim = min((x0+dx, max(self.x)))
            low_lim = max((x0-dx, min(self.x)))
            logger.warning(' Spectrum does not cover full range used for determining normalization constant. Spectral range used: [{low}:{high}]'.format(low=repr(low_lim), high=repr(high_lim)))

        if correct_flux:
            fluxz = correct_flux_(self.x, flux, self.z, z0)
            flux = fluxz

#        try:
#            avg_flux = numpy.mean(flux[idx[0]:idx[1]])
#        except FloatingPointError:
#            avg_flux = flux[idx[0]]

        try:
            avg_flux = numpy.mean(flux[spec_indices[0]:spec_indices[1]])
        except FloatingPointError:
            avg_flux = flux[spec_indices[0]]

        if norm_operator == 0:
            norm_constant = y0 / avg_flux
            norm_flux = flux*norm_constant
            norm_fluxerr = fluxerr*norm_constant if self.yerr is not None else None
        elif norm_operator == 1:
            norm_constant = y0 - avg_flux
            norm_flux = flux + norm_constant
            norm_fluxerr = fluxerr
        else:
            raise ValueError('Unrecognized norm_operator. keyword \'norm_operator\' must be either 0 (to multiply) or 1 (to add)')

        norm_spectrum = Spectrum(x=self.x, y=numpy.array(norm_flux), yerr=norm_fluxerr,
                                 xunit=self.xunit, yunit=self.yunit, z=self.z)

        setattr(norm_spectrum, 'norm_constant', norm_constant)

        # keep attributes of old sed
        _get_setattr(norm_spectrum,self)

        return norm_spectrum


    def normalize_by_int(self, minWavelength='min', maxWavelength='max', correct_flux=False, z0=None):
        '''Normalizes the Spectrum such that the area under the specified wavelength range is equal to 1.

        Algorithm taken from astLib.astSED.normalise(); uses the Trapezoidal rule to estimate the integrated flux.

        Parameters
        ----------
        minWavelength : float or 'min'
            Minimum wavelength of range over which to normalize Spectrum
        maxWavelength : float or 'max'
            Maximum wavelength of range over which to normalize Spectrum
        correct_flux : bool
            Switch used to correct for SEDs that were shifted to some redshift without taking into account flux brightening/dimming due to redshift
        z0 : float or int, optional
            The original redshift of the source. Used if correct_flux = True.

        Raises
        ------
        sedstacker.exceptions.SegmentError
            If the Spectrum object has less than 2 points between `minWavelength` and `maxWavelength`.')

        Notes
        -----
        The Spectrum must have at least 2 points between minWavelength and maxWavelength.

        Examples
        --------
        >>> # initializing dummy spectrum
        >>> 
        >>> x = numpy.arange(3000, 9500, 0.5)
        >>> y = numpy.linspace(1, 1000, x.size)
        >>> yerr = y*0.01
        >>> spec = Spectrum(x=x,y=y,yerr=yerr,z=0.3)
        >>> 
        >>> # normalize the spectrum over its full range
        >>> 
        >>> norm_spec = spec.normalize_by_int()
        >>> norm_spec.y
        >>> array([  3.07455874e-07,   3.31084493e-07,   3.54713112e-07, ...,
                     3.07408617e-04,   3.07432246e-04,   3.07455874e-04])
        >>> 
        >>> # Check the normalization constant
        >>> 
        >>> norm_spec.norm_constant
        3.0745587412510564e-07

        '''

        flux = numpy.ma.masked_invalid(self.y)
        fluxerr = numpy.ma.masked_invalid(self.yerr)

        if minWavelength == 'min':
            minWavelength=self.x.min()
        if maxWavelength == 'max':
            maxWavelength=self.x.max()

        # Check that minWavelength is shorter than maxWavelength
        if minWavelength >=maxWavelength:
            raise BadRangesError("The min wavelength must be shorter than the max wavelength.")

        lowCut = numpy.greater(self.x, minWavelength)
        highCut = numpy.less(self.x, maxWavelength)
        totalCut = numpy.logical_and(lowCut, highCut)
        sedWavelengthSlice = self.x[totalCut]

        if len(sedWavelengthSlice) < 2:
            raise SegmentError('Spectrum object must have at least 2 points between minWavelength and maxWavelength.')

        if correct_flux:
            fluxz = correct_flux_(self.x, flux, self.z, z0)
            flux = fluxz

        sedFluxSlice = flux[totalCut]
        norm_constant = 1.0/numpy.trapz(abs(sedFluxSlice), sedWavelengthSlice)
        norm_flux = numpy.array(flux*norm_constant)
        norm_fluxerr = numpy.array(fluxerr*norm_constant if fluxerr is not None else None)
        norm_segment = Spectrum(x=self.x, y=norm_flux, yerr=norm_fluxerr,
                                xunit=self.xunit, yunit=self.yunit, z=self.z)

        setattr(norm_segment, 'norm_constant', norm_constant)

        # keep attributes of old sed
        _get_setattr(norm_segment,self)

        return norm_segment


    def write(self, filename, fmt='ascii'):
        '''Write Spectrum to file.

        Parameters
        ----------
        filename : str
            The name of the output file.
        fmt : str
            The format for the output file. The default file format is 'ascii'. For release 1.0, only ASCII files will be supported.

        Examples
        --------
        >>> # Writing a Spectrum object with no flux errors
        >>> # (i.e. spectrum.yerr = None)
        >>> 
        >>> spectrum.write('my_data_directory/sed_data.txt')
        >>> more my_data_directory/sed_data.txt
        x y
        1941.8629 0.046853197
        1942.5043 0.059397754
        1943.1456 0.032893488
        1943.7870 0.058623008
        ...            ... 
        10567.7890 0.046843890
        10568.4571 0.059888754

        '''

        if os.path.exists(filename):
            raise PreExistingFileError(filename)
        else:
            segment_arrays = [self.x, self.y, self.yerr]
            ascii.write(segment_arrays, filename, names=['x','y','y_err'], comment='#')


class Sed(Segment, list):
    '''Represents a photometric SED from one astronomical object or model.

    Parameters
    ----------
    x : array_like
        The spectral coordinates. Default value is an empty list, [].
    y : array_like
        The flux values. Default value is an empty list, [].
    yerr : array_like of float or int; or None
        The errors on the flux values. Default value is None.
    xunit : array_like of str
        The spectral coordinate units. Default value is ['AA'].
    yunit : array_like of str
        The flux coordinates. Default value is ['erg/s/cm**2/AA'].
    z : float, int
        The redshift of the Sed. Default value is None.
    
    Raises
    ------
    sedstacker.exceptions.SegmentError
        If *x*, *y*, *yerr*, *xunit*, and/or *yunit* do not have the same length.
    
    Notes
    -----
    - x and y must have the same length.
    - If all y-values share the same error, yerr can be a float or integer; each point (x,y) will be assigned that value as *yerr*. To  do this, set ``yerr=[yerr_value]``.
    - If all spectral coordinates *x* have the same units, then set ``xunit=['unit_name']``.
    - If all flux coordinates *y* have the same units, then set ``yunit=['unit_name']``.
    
    Examples
    --------
    >>> sed = Sed(x=[1212.0, 3675.0, 4856.0], y=[1.456e-11, 3.490e-11, 5.421e-11], yerr=1.0e-13, z=0.02)
    >>> sed.yerr
    1.0e-13
    1.0e-13
    1.0e-13

    Attributes
    ----------
    x : numpy.array of floats
        The spectral coordinates.
    y : numpy.array of floats
        The flux values.
    yerr : numpy.array of floats
        The errors on the flux values.
    xunit : numpy.array of str
        The spectral coordinate units.
    yunit : numpy.array of str
        The flux coordinates.
    z : float, int
        The redshift of the Sed.
    norm_constant : float
        The normalization constant of the Sed. Default value is None.
    counts : array of int
        Array of the number of points combined in each bin through ``sedstacker.sed.stack()``. Default value is None.

    '''

    def __init__(self, x=[], y=[], yerr=None, xunit=['AA'], yunit=['erg/s/cm**2/AA'], z=None):
        
        self.z = z
        self.counts = None
        self.norm_constant = 1.0
        
        if len(x) != len(y):
            raise SegmentError('x and y must be of the same length.')

        if yerr is None:
            yerr = [numpy.nan]*len(x)
        else:
            if len(yerr) == 1:
                yerr = [yerr]*len(x)
            elif len(yerr) != len(x):
                raise SegmentError('x and yerr must be of the same length.')

        if isinstance(xunit,types.NoneType):
            xunit = [xunit]*len(x)
        elif len(xunit) == 1:
            xunit = xunit*len(x)
        elif len(xunit) != len(x):
            raise SegmentError('xunit must have the same length as x.')

        if isinstance(yunit,types.NoneType):
            yunit = [yunit]*len(y)
        elif len(yunit) == 1:
            yunit = yunit*len(y)
        elif len(yunit) != len(y):
            raise SegmentError('yunit must have the same length as y.')
        
        for i in range(len(x)):
            point = PhotometricPoint(x=x[i], y=y[i], yerr=yerr[i], xunit=xunit[i], yunit=yunit[i])
            self.append(point)

    @property
    def x(self):
        return self._toarray()[0]
    @x.setter
    def x(self, val):
        assert len(val) == len(self), 'x array and Sed object must have same length.'
        for i, point in enumerate(self):
            point.x = val[i]
    @x.deleter
    def x(self):
        raise AttributeError('Cannot delete property x.')
    
    @property
    def y(self):
        return self._toarray()[1]
    @y.setter
    def y(self, val):
        assert len(val) == len(self), 'x array and Sed object must have same length.'
        for i, point in enumerate(self):
            point.y = val[i]
    @y.deleter
    def y(self):
        raise AttributeError('Cannot delete attribute y.')

    @property
    def yerr(self):
        return self._toarray()[2]
    @yerr.setter
    def yerr(self, val):
        #Sets flux-error, yerr. If val is a single number, all fluxerrors are assigned val. If val is an iterable, then each point is assigned the consecutive values in val.
        try:
            assert len(val) == len(self), 'yerr array and y must have same length.'
            for i, point in enumerate(self):
                assert type(val[i]) in NUMERIC_TYPES, 'yerr must be of numeric type'
                point.yerr = val[i]
        except TypeError:
            assert type(val) in NUMERIC_TYPES, 'yerr must be of numeric type'
            for point in self:
                point.yerr = val
    @yerr.deleter
    def yerr(self):
        logging.info('Setting yerr to None.')
        self._yerr=None

    @property
    def xunit(self):
        return self._toarray()[3]
    @xunit.setter
    def xunit(self, val):
        #Sets all x-unit values to val
        if not isinstance(val, types.StringType):
            raise TypeError('val must be a string.')
        for point in self:
            point.xunit = val
    @xunit.deleter
    def xunit(self):
        logging.info('Setting xunit to None.')
        self._xunit=None

    @property
    def yunit(self):
        return self._toarray()[4]
    @yunit.setter
    def yunit(self, val):
        #Sets all y-unit values to val
        if not isinstance(val, types.StringType):
            raise TypeError('val must be a string.')
        for point in self:
            point.yunit = val
    @yunit.deleter
    def yunit(self):
        logging.info('Setting yunit to None.')
        self._yunit=None

    @property
    def z(self):
        return self._z
    @z.setter
    def z(self, val):
        if isinstance(val, types.NoneType):
            self._z = val
        elif type(val) not in NUMERIC_TYPES:
            raise InvalidRedshiftError(0)
        elif val < 0:
            raise InvalidRedshiftError(1)
        else:
            self._z = numpy.float_(val)
    @z.deleter
    def z(self):
        logging.info('Setting z to None.')
        self._z = None


    def __str__(self):
        data = Table([self.x, self.y, self.yerr, self.xunit, self.yunit], names=('x','y','yerr','xunit','yunit'))
        return data.__str__()


    def shift(self, z0, correct_flux=True):
        '''Redshifts the spectrum by means of cosmological expansion.
        
        Parameters
        ----------
        z0 : float, int
            Target redshift to shift the SED/spectrum to
        correct_flux : bool
            If True, the flux will be corrected for the intrinsic dimming/brightening due to shifting the spectrum. If False, only the spectral coordinates will be shifted; the flux remains the same.

        Returns
        -------
        sedstacker.sed.Sed
            A new Sed object with the redshifted SED.

        Raises
        ------
        sedstacker.exceptions.InvalidRedshiftError
            If attribute *z* or parameter *z0* is of non-numeric type or is negative.
        sedstacker.exceptions.NoRedshiftError
            If attribute *z* is *None*.

        Examples
        --------
        >>> # Shift the SED to rest frame
        >>> 
        >>> rf_spectrum = sed.shift(0)
        >>>
        >>> # Shift the spectrum to rest frame, without correcting the flux
        >>> 
        >>> rf_spectrum = sed.shift(0, correct_flux=False)

        '''

        if isinstance(self.z, types.NoneType):
            raise NoRedshiftError
        if (not type(z0) in NUMERIC_TYPES) or (not type(self.z) in NUMERIC_TYPES):
            raise InvalidRedshiftError(0)
        if z0 < 0.0 or self.z < 0.0:
            raise InvalidRedshiftError(1)

        if correct_flux:
            spec_z0, flux_z0 = shift(self.x, self.y, self.z, z0)
        else:
            spec_z0 = (1 + z0) * self.x / (1+self.z)
            flux_z0 = self.y

        spec = Sed(x=spec_z0, y=flux_z0, yerr=self.yerr,
                        xunit=self.xunit, yunit=self.yunit, z=z0)
        # keep attributes of old sed
        _get_setattr(spec,self)

        return spec


    def normalize_at_point(self, x0, y0, norm_operator=0, correct_flux=False, z0=None):
        '''Normalizes the SED such that at spectral coordinate x0, the flux of the SED is y0.

        Notes
        -----
        Uses nearest-neighbor interpolation to calculate the normalization constant.

        Parameters
        ----------
        x0 : float or int
            The spectral coordinate to normalize the SED at
        y0 : float or int
            The flux value to normalize the SED to
        correct_flux : bool
            To correct for flux dimming/brightening due to redshift. Meant for SEDs that were shifted only by wavelength (i.e. the flux was not corrected for the intrinsic dimming/brightening due to redshift). If ``correct_flux = True``, then the flux is corrected so that the integrated flux at the current redshift is equal to that at the original redshift.Default value is False.
        norm_operator : int
            Operator used for scaling the spectrum to y0.
                - 0 = multiply the flux by the normalization constant
                - 1 = add the normalization constant to the flux
        z0 : float or int, optional
            The original redshift of the source. Used only if ``correct_flux = True``.

        Returns
        -------
        sedstacker.sed.Sed
            A new Sed object of the normalized SED.

        Raises
        ------
        exceptions.ValueError
            If an unrecognized *norm_operator* value is used.

        Examples
        --------
        >>> # initializing dummy SED
        >>> 
        >>> from numpy import logspace, linspace, log10
        >>> x = logspace(log10(3000), log10(70000), num=20)
        >>> y = linspace(1, 1000, num=x.size)*1e-5
        >>> yerr = y*0.01
        >>> sed = Sed(x=x,y=y,yerr=yerr,z=0.3)
        >>>
        >>> # normalize the SED at 6000.0 Angstroms and 1e-3 erg/s/cm**2/Angstrom
        >>> 
        >>> norm_sed = sed.normalize_at_point(6000, 1e-3, norm_operator=0)
        >>>
        >>> norm_sed.y
        array([  4.73225405e-06,   2.53549191e-04,   5.02366127e-04,
                 7.51183064e-04,   1.00000000e-03,   1.24881694e-03,
                 1.49763387e-03,   1.74645081e-03,   1.99526775e-03,
                 2.24408468e-03,   2.49290162e-03,   2.74171856e-03,
                 2.99053549e-03,   3.23935243e-03,   3.48816936e-03,
                 3.73698630e-03,   3.98580324e-03,   4.23462017e-03,
                 4.48343711e-03,   4.73225405e-03])
        >>>
        >>> # Checking the normalization constant
        >>> 
        >>> norm_sed.norm_constant
        0.473225404732254

        '''

        spec = self.x
        flux = self.y
        fluxerr = self.yerr
        xunit = self.xunit
        yunit = self.yunit

        if correct_flux:
            fluxz = correct_flux_(spec, flux, self.z, z0)
            flux = fluxz  

        interp_self = calc.fast_nearest_interp([x0], spec, flux)
        flux_at_x0 = numpy.float_(interp_self)

        if norm_operator == 0:
            norm_constant = y0 / flux_at_x0
            norm_flux = flux * norm_constant
            norm_fluxerr = fluxerr * norm_constant if fluxerr is not None else None
        elif norm_operator == 1:
            norm_constant = y0 - flux_at_x0
            norm_flux = flux + norm_constant
            norm_fluxerr = fluxerr
        else:
            raise ValueError('Unrecognized norm_operator. keyword \'norm_operator\' must be either 0 (for multiply) or 1 (for addition).')

        norm_sed = Sed(x=spec, y=norm_flux, yerr=norm_fluxerr, xunit=xunit, yunit=yunit, z=self.z)
        norm_sed.norm_constant = norm_constant
        # keep attributes of old sed
        _get_setattr(norm_sed,self)

        return norm_sed
    

    def normalize_by_int(self, minWavelength='min', maxWavelength='max', correct_flux=False, z0=None):

        '''Normalises the SED such that the area under the specified wavelength range is equal to 1.

        Notes
        -----
        Algorithm adopted from astLib.astSED.normalise(); uses the Trapezoidal rule to estimate the integrated flux.

        Parameters
        ----------
        minWavelength : float or 'min'
            Minimum wavelength of range over which to normalise SED
        maxWavelength : float or 'max'
            Maximum wavelength of range over which to normalise SED
        correct_flux : bool
            Switch used to correct for SEDs that were shifted to some redshift without taking into account flux brightening/dimming due to redshift
        z0 : float or int, optional
            The original redshift of the source. Used if correct_flux = True.

        The SED must have at least 2 points between minWavelength and maxWavelength.

        Returns
        -------
        sedstacker.sed.Sed
            A new Sed object of the normalized SED.

        Raises
        ------
        sedstacker.exceptions.SegmentError
            If the Sed has less than two points between *minWavelength* and *maxWavelength*.

        Examples
        --------
        >>> # initializing dummy SED
        >>> 
        >>> from numpy import logspace, linspace, log10
        >>> x = logspace(log10(3000), log10(70000), num=20)
        >>> y = linspace(1, 1000, num=x.size)*1e-5
        >>> yerr = y*0.01
        >>> sed = Sed(x=x,y=y,yerr=yerr,z=0.3)
        >>> 
        >>> # normalize the SED over its full range
        >>> norm_sed = sed.normalize_by_int()
        >>> 
        >>> norm_sed.y
        array([  2.05343055e-08,   1.10020647e-06,   2.17987864e-06,
                 3.25955081e-06,   4.33922298e-06,   5.41889515e-06,
                 6.49856732e-06,   7.57823949e-06,   8.65791166e-06,
                 9.73758383e-06,   1.08172560e-05,   1.18969282e-05,
                 1.29766003e-05,   1.40562725e-05,   1.51359447e-05,
                 1.62156168e-05,   1.72952890e-05,   1.83749612e-05,
                 1.94546333e-05,   2.05343055e-05])
        >>>
        >>> # Check the normalization constant
        >>> 
        >>> norm_sed.norm_constant
        0.0020534305518967668

        '''

        spec = self.x
        flux = numpy.ma.masked_invalid(self.y)
        fluxerr = numpy.ma.masked_invalid(self.yerr)
        xunit = self.xunit
        yunit = self.yunit

        if minWavelength == 'min':
            minWavelength=spec.min()
        if maxWavelength == 'max':
            maxWavelength=spec.max()

        # Check that minWavelength is shorter than maxWavelength
        if minWavelength >=maxWavelength:
            raise ValueError("The min wavelength must be shorter than the max wavelength.")

        lowCut = numpy.greater_equal(spec, minWavelength)
        highCut = numpy.less_equal(spec, maxWavelength)
        totalCut = numpy.logical_and(lowCut, highCut)
        sedWavelengthSlice = spec[totalCut]

        if len(sedWavelengthSlice) < 2:
            raise SegmentError('Sed object must have at least 2 points between minWavelength and maxWavelength.')

        if correct_flux:
            fluxz = correct_flux_(spec, flux, self.z, z0)
            flux = fluxz

        sedFluxSlice = flux[totalCut]
        norm_constant = 1.0/numpy.trapz(abs(sedFluxSlice), sedWavelengthSlice)
        norm_flux = numpy.array(flux*norm_constant)
        norm_fluxerr = numpy.array(fluxerr*norm_constant if fluxerr is not None else None)

        norm_segment = Sed(x=spec, y=norm_flux, yerr=norm_fluxerr,
                           xunit=xunit, yunit=yunit, z=self.z)
        norm_segment.norm_constant = norm_constant
        # keep attributes of old sed
        _get_setattr(norm_segment,self)

        return norm_segment


    def add_point(self, point):
        '''
        Add a PhotometricPoint to Sed object.

        Parameters
        ----------
        point : sedstacker.sed.PhotometricPoint
            A PhotometricPoint object.

        Examples
        --------
        >>> # Set up an SED
        >>> 
        >>> x = [12491.0, 21590.4]
        >>> y = [20.81, 20.39]
        >>> yerr = [0.09, numpy.nan]
        >>> sed = Sed(x=x, y=y, yerr=yerr, xunit='AA', yunit='mag', z=0.92)
        >>> print sed
           x       y   yerr xunit yunit
        -------- ----- ---- ----- -----
         12491.0 20.81 0.09    AA   mag
         21590.4 20.39  nan    AA   mag
        >>> 
        >>> # Add a new point
        >>> 
        >>> point = PhotometricPoint(x=36000.0, y=19.78, yerr=0.09, xunit="AA", yunit='mag')
        >>> sed.add_point(point)
        >>> print sed
           x       y   yerr xunit yunit
        -------- ----- ---- ----- -----
         12491.0 20.81 0.09    AA   mag
         21590.4 20.39  nan    AA   mag
         36000.0 19.78 0.09    AA   mag  # new point

        '''
        if isinstance(point, PhotometricPoint):
            self.append(point)
        else:
            raise TypeError('Only PhotometricPoints may be added to a Sed object.')


    def remove_point(self, index):
        '''Remove a PhotometricPoint from Sed object.

        Parameters
        ----------
        index : int
            The index of the PhotometricPoint in the list to remove.

        Examples
        --------
        >>> # Set up an SED
        >>> 
        >>> x = [12491.0, 21590.4, 36000.0]
        >>> y = [20.81, 20.39, 19.78]
        >>> yerr = [0.09, numpy.nan, 0.09]
        >>> sed = Sed(x=x, y=y, yerr=yerr, xunit='AA', yunit='mag', z=0.92)                                                                 
        >>> print sed
           x       y   yerr xunit yunit
        -------- ----- ---- ----- -----
         12491.0 20.81 0.09    AA   mag
         21590.4 20.39  nan    AA   mag # remove this point
         36000.0 19.78 0.09    AA   mag
        >>> 
        >>> sed.remove_point(1)
        >>> print sed
           x       y   yerr xunit yunit
        -------- ----- ---- ----- -----
         12491.0 20.81 0.09    AA   mag
         36000.0 19.78 0.09    AA   mag

        '''

        self.pop(index)


    def mask_point(self, index):
        '''Mask a point. 

        Parameters
        ----------
        index : int
            The index of the PhotometricPoint in the list to mask.

        Examples
        --------
        >>> sed.mask_point(1)

        '''

        
        self[index].mask = True


    def unmask_point(self, index):
        '''Unmask a point. 
        Parameters
        ----------
        index : int
            The index of the PhotometricPoint in the list to unmask.

        Examples
        --------
        >>> sed.unmask_point(1)

        '''
        
        self[index].mask = False


    def add_segment(self, *segments):
        '''Add another Sed or iterable of Seds to a Sed. 

        Parameters
        ----------
        segments : array-like of sedstacker.sed.Sed's
            The SEDs to add to Sed.

        Examples
        --------
        >>> sed1 = 
        >>> sed2 = 
        >>> sed1.add_segment(sed2)

        '''
        
        for segment in segments:
            for points in segment:
                self.append(points)


    def _toarray(self):
        '''Convert a Sed to a 5-dimensional tuple of arrays of x, y, yerr, xunit and yunit.

        For example, if

        >>> sedarray = Sed()._toarray()

        then::
            sedarray[0] --> spectral axis, 'x'
            sedarray[1] --> flux axis, 'y'
            sedarray[2] --> flux-error axis, 'yerr'
            sedarray[3] --> xunit values, 'xunit'
            sedarray[4] --> yunit values, 'yunit'
        
        Returns
        -------
        5-dimensional array
            The spectral, flux, flux-error, xunit and yunit values in the Sed.

        '''
        
        spec = numpy.array([p.x for p in self if p.mask == False])
        flux = numpy.array([p.y for p in self if p.mask == False])
        fluxerr = numpy.array([p.yerr for p in self if p.mask == False])
        xunit = numpy.array([p.xunit for p in self if p.mask == False])
        yunit = numpy.array([p.yunit for p in self if p.mask == False])

        sedarray = spec, flux, fluxerr, xunit, yunit

        return sedarray


    def write(self, filename, xunit='AA', yunit='erg/s/cm**2/AA', fmt='ascii'):
        '''Write Sed to file.

        Parameters
        ----------
        filename : str
            The name of the output file.
        fmt : str
            The format for the output file. The default file format is 'ascii'. For release 1.0, only ASCII files will be supported.

        Examples
        --------

        >>> sed.write('my_data_directory/sed_data.txt')
        >>> more my_data_directory/sed_data.txt
        x y
        1941.8629 0.046853197
        1942.5043 0.059397754
        1943.1456 0.032893488
        1943.7870 0.058623008
        ...            ... 
        10567.7890 0.046843890
        10568.4571 0.059888754

        '''

        if os.path.exists(filename):
            raise PreExistingFileError(filename)
        else:
            sed = self._toarray()
            if self.counts is not None:
                segment_arrays = Table({'x':sed[0],
                                        'y':sed[1],
                                        'y_err':sed[2],
                                        'counts':self.counts},
                                       names=['x','y','y_err','counts'],
                                       dtype=('f10','f10','f10','i5')
                                       )
                ascii.write(segment_arrays, filename, comment='#', names=['x','y','y_err', 'counts'])
            else:
                segment_arrays = [sed[0],sed[1],sed[2]]
                ascii.write(segment_arrays, filename, names=['x','y','y_err'], comment='#')


class Stack(list):
    ''' A collection of Sed's and/or Spectra to stack. Users can normalize and redshift all the segments in an Stack simultaneously. 

    Attributes
    ----------
    x : list of numpy.array's
        The spectral values for each segment.
    y : list of numpy.array's
        The flux values for each segment
    yerr : list of numpy.array's
        The flux-errors for each segment
    xunit : list of numpy.array's
        The spectral units for each segment.
    yunit : list of numpy.array's
        The flux units for each segment.
    z : list of float and/or int
        The redshifts of the segments.
    segments : list
        The sedstacker.sed.Sed and sedstacker.sed.Spectrum objects in the Stack.

    Notes
    -----
    Attributes *x*, *y*, *yerr*, *xunit* and *yunit* are *M x N\ :sub:i\ * arrays, where *M* is the number of Seds/Spectra in the Stack, and N\ :sub:i\ is the number of points in the Sed or Spectrum.

    '''
    def __init__(self, segments):

        self.segments = segments

        self.x = []
        self.y = []
        self.yerr = []
        self.xunit = []
        self.yunit = []
#        self.z = []

        for segment in segments:
            if not isinstance(segment, Segment):
                raise NotASegmentError
            else:
                self.x.append(segment.x)
                self.y.append(segment.y)
                self.yerr.append(segment.yerr)
                self.xunit.append(segment.xunit)
                self.yunit.append(segment.yunit)
#                self.z.append(segment.z)
                self.append(segment)

        # If I want to concentate all the spec and flux arrays:
        # [in the for loop: count += len(segment.x)]
        
        #        self.x.reshape(1,count)
        #        self.y.reshape(1,count)
        #        self.yerr.reshape(1,count)
        #        self.xunit.reshape(1,count)
        #        self.yunit.reshape(1,count)


    def __str__(self):
        pass

    def shift(self, z0, correct_flux=True):
        '''
        Redshifts the Seds and/or Spectra in the Stack by means of cosmological
        expansion.

        Args:
            z0 (float, int): Target redshift to shift the Stack to
        
        Kwargs:
            correct_flux (bool): If True, the flux will be corrected for the intrinsic
            dimming/brightening due to shifting the spectrum.
            If False, only the spectral coordinates will be shifted; the flux remains
            the same.

        Raises:
            NoRedshift: Raised if a Sed/Spectrum has no redshift. If raised, the segment
            is excluded from the returned shifted Stack.
            InvalidRedshiftError: Raised if z0 is not of numeric type Float or Integer, or if z0 is negative. If raised, the segment is excluded from the returned shifted Stack.
        
        Returns:
            A new Stack object with the redshifted SED.

        Ex:
        
        >>> # group 6 segments into an Stack,
        >>> # then shift them all to z=1.0.
        >>> aggsed = Stack([sed1, sed2, sed3, sed4, spec1, spec2])
        >>> aggsed_z1 = aggsed.shift(1)

        If 'sed2' had no redshift (i.e. sed2.z = None) and we shift the Stack, a warning appears, stating the index of the segment that is excluded from the shifted Stack.

        >>> aggsed_z1 = aggsed.shift(1)
        WARNING: Excluding Stack[1] from the shifted Stack.

        '''

        shifted_segments = []

        for segment in self.segments:
            try:
                shifted_seg = segment.shift(z0, correct_flux = correct_flux)
                shifted_segments.append(shifted_seg)
            except NoRedshiftError:
                logger.warning(' Excluding Stack[%d] from the shifted Stack.' % self.index(segment))
                pass

            except InvalidRedshiftError:
                logger.warning(' Excluding Stack[%d] from the shifted Stack.' % self.index(segment))
                pass

        return Stack(shifted_segments)


    def filter(self, boolean = '>', **kwargs):
        raise NotImplemented('filter() is not implemented yet.')


    def normalize_at_point(self, x0, y0, norm_operator=0, correct_flux=False, z0=None):
        '''Normalizes the SEDs such that at spectral coordinate x0, the flux of the SEDs is y0. Uses the parent class :func: normalize_at_point() method (i.e. sedstacker.sed.Sed.normalize_at_point() is used on Sed objects in the Stack, while sedstacker.sed.Spectrum.normalize_at_point() is used on Spectrum objects)

        Parameters
        ----------
        x0 : float or int
            The spectral coordinate to normalize the SEDs at
        y0 : float or int
            The flux value to normalize the SEDs to
        correct_flux : bool
            Correct for flux dimming/brightening due to redshift. Meant for SEDs that were shifted only by wavelength (i.e. the flux was not corrected for the intrinsic dimming/brightening due to redshift). If ``correct_flux = True``, then the flux is corrected so that the integrated flux at the current redshift is equal to that at the original redshift. Default value is *False*.
        norm_operator : int
            operator used for scaling the spectrum to *y0*.
            - 0 = multiply the flux by the normalization constant
            - 1 = add the normalization constant to the flux
        z0 : float or int, optional
            A list or array of the original redshifts of the sources. The redshifts must be in the same order as the Segments appear in the Stack (i.e. assuming aggsed is an Stack, z0[0] is the original redshift of aggsed[0], z0[1] is the original redshift of aggsed[1], etc.). Used only if correct_flux = True.

        Notes
        -----
        - If a Spectrum's spectral range does not cover point x0, then the segment is excluded from the returned normalized Stack.
        - If a Spectrum has less than 4 points, then the Sed or Spectrum is excluded from the returned normalized Stack.
        
        Raises
        ------
        exceptions.ValueError
            If the length of *z0* does not equal the number of segments in the Stack.

        Examples
        --------
        Usage is the same as it is for Sed and Spectrum objects. Let's say we have instantiated 6 segments (4 SEDs and 2 spectra) that we want to normalize together:

        >>> # group 6 segments into an Stack,
        >>> # then normalize them at point (5000 AA, 1 erg/s/cm**2/AA)
        >>> 
        >>> aggsed = Stack([sed1, sed2, sed3, sed4, spec1, spec2])
        >>> norm_seds = seds.normalize_at_point(5000, 1)

        We can view the normalization constants for each segments in normalized Stack. For example, to view the normalization constant of 'sed2':

        >>> norm_seds[1].norm_constant
        0.473225404732254

        If 'sed2' has no data at 5000 Angstroms, 'sed2' will not be normalized and will be exluded from 'norm_seds'. A warning states the index of the segment that is excluded from the shifted Stack.

        >>> norm_seds = seds.normalize_at_point(5000, 1)
        WARNING: Excluding Stack[1] from the normalized Stack.

        '''

        if isinstance(z0,(types.FloatType, types.IntType, numpy.float_,numpy.int_,types.NoneType)):
            z0 = [z0]*len(self.segments)
        elif len(z0) != len(self.segments):
            raise ValueError('Length of z0 does not match the length of Stack.')

        norm_segments = []

        for i, segment in enumerate(self.segments):
            try:
                norm_seg = segment.normalize_at_point(x0, y0,
                                                      norm_operator=norm_operator,
                                                      correct_flux=correct_flux,
                                                      z0=z0[i])
                norm_segments.append(norm_seg)
            except OutsideRangeError:
                logger.warning(' Excluding Stack[%d] from the normalized Stack.' % self.index(segment))
                pass
            except SegmentError, e:
                logger.warning(' Excluding Stack[%d] from the normalized Stack' % self.index(segment))
                pass

        norm_segments = Stack(norm_segments)

        return norm_segments


    def normalize_by_int(self, minWavelength='min', maxWavelength='max', correct_flux=False, z0=None):
        '''Normalises the SEDs such that the area under the specified wavelength range is equal to 1.

        Parameters
        ----------
        minWavelength : float or 'min'
            Minimum wavelength of range over which to normalise SED
        maxWavelength : float or 'max'
            Maximum wavelength of range over which to normalise SED
        correct_flux : bool
            Switch used to correct for SEDs that were shifted to some redshift without taking into account flux brightening/dimming due to redshift
        z0 : array_like, optional
            A list or array of the original redshifts of the sources. The redshifts must be in the same order as the Segments appear in the Stack (i.e. assuming aggsed is an Stack, ``z0[0]`` is the original redshift of ``aggsed[0]``, ``z0[1]`` is the original redshift of ``aggsed[1]``, etc.). Used only if ``correct_flux = True``.

        Notes
        -----
        - If a Sed or Spectrum has less than 2 points within minWavelength and maxWavelength, then the Sed or Spectrum is excluded from the returned normalized Stack.

        Examples
        --------
        Usage is the same as it is for Sed and Spectrum objects. Let's say we have instantiated 6 segments (4 SEDs and 2 spectra) that we want to normalize by the integrated flux at optical wavelengths, 3000 AA to 10,000 AA:

        >>> # group 6 segments into an Stack,
        >>> # then normalize them together
        >>>
        >>> aggsed = Stack([sed1, sed2, sed3, sed4, spec1, spec2])
        >>> norm_seds = seds.normalize_by_int(minWavelength=3000, maxWavelength=10000)

        We can view the normalization constants for each segments in normalized Stack. For example, to view the normalization constant of 'sed2':

        >>> norm_seds[1].norm_constant
        0.473225404732254

        If 'spec1' has no data within the range (*minWavelength, maxWavelength*), then 'spec1' will not be normalized and will be exluded from 'norm_seds'. A warning states the index of the segment that is excluded from the shifted Stack.

        >>> norm_seds = seds.normalize_by_int(minWavelength=9000, maxWavelength=10000)
        WARNING: Excluding Stack[4] from the normalized Stack.

        '''

        if isinstance(z0, NUMERIC_TYPES+(types.NoneType,)):
            z0 = [z0]*len(self.segments)
        elif len(z0) != len(self.segments):
            raise ValueError('Length of z0 does not match length of Stack.')

        norm_segments = []

        for i, segment in enumerate(self.segments):
            try:
                norm_seg = segment.normalize_by_int(minWavelength=minWavelength,
                                                    maxWavelength=maxWavelength,
                                                    correct_flux=correct_flux,
                                                    z0 = z0[i])
                norm_segments.append(norm_seg)
            except SegmentError:
                logger.warning(' Excluding Stack[%d] from the normalized Stack' % self.index(segment))
                pass

        norm_segments = Stack(norm_segments)
        return norm_segments


    def add_segment(self, segment):
        '''Add a segment to the Stack.

        Parameters
        ----------
        segment : sedstacker.sed.Segment
            A sedstacker.sed.Sed or sedstacker.sed.Spectrum.

        Examples
        --------

        >>> seds = Stack([sed1,sed2,sed3,sed4,spec1,spec2])
        >>> seds.add_segment(spec3)
        >>> len(seds)
        7
        '''

        #        if isinstance(segments, types.ListType):
        #            for segment in segments:
        #                if isinstance(segment, Segment):
        #                    self.append(segment)
        #                    self.segments.append(segment)
        #            else:
        #                raise NotASegmentError
        #        else:
        if isinstance(segment, Segment):
            self.append(segment)
            self.segments.append(segment)
            #self.z.append(segment.z)
            self.x.append(segment.x)
            self.y.append(segment.y)
            self.yerr.append(segment.yerr)
            self.xunit.append(segment.xunit)
            self.yunit.append(segment.yunit)
        else:
            raise NotASegmentError


    def remove_segment(self, segment):
        '''Remove a segment from the Stack.

        Parameters
        ----------
        segment : sedstacker.sed.Segment
            A sedstacker.sed.Sed or sedstacker.sed.Spectrum.

        Examples
        --------

        Let's say we have instantiated 6 segments (4 SEDs and 2 spectra), then added them to an Stack. We wish to remove the second SED from the Stack.

        >>> seds = Stack([sed1,sed2,sed3,sed4,spec1,spec2])
        >>> len(seds)
        6
        >>> # Remove sed2.
        >>> 
        >>> seds.remove_segment(sed2)
        >>> len(seds)
        5

        '''

        index = self.index(segment)
        self.x.pop(index)
        self.y.pop(index)
        self.yerr.pop(index)
        self.xunit.pop(index)
        self.yunit.pop(index)
        #self.z.pop(index)

        self.remove(segment)
        self.segments.remove(segment)


    def write(self, filename, xunit='AA', yunit='erg/s/cm**2/AA', fmt='ascii'):
        '''Write an Stack to file.

        The Seds and Spectra are written to file one after the other, in the order they are indexed in the Stack.

        Parameters
        ----------
        filename : str
            The name of the output file.
        fmt : str
            The format for the output file. The default file format is 'ascii'. For release 1.0, only ASCII files will be supported.

        Examples
        --------
        Let's say we have instantiated 6 segments (4 SEDs and 2 spectra), then added them to an Stack. We wish to write the data to an ASCII file:

        >>> seds = Stack([sed1,sed2,sed3,sed4,spec1,spec2])
        >>> seds.write('my_data_directory/sed_data.txt')
        >>> more my_data_directory/sed_data.txt
        x y
        3823.0 0.0424           # first SED
        4459.7 0.0409
        5483.8 0.0217
        4779.6 0.0345
        ...    
        80000.0 0.912
        240000.0 1.245
        1941.8629 0.046853197   # second SED
        1942.5043 0.059397754
        1943.1456 0.032893488
        1943.7870 0.058623008
        ...       
        10567.7890 0.046843890
        10568.4571 0.059888754
        ...                    # and so on
            

        '''

        if os.path.exists(filename):
            raise PreExistingFileError(filename)
        else:
            segment_x = []
            segment_y = []
            segment_yerr = []
            counts = []
            for i in range(len(self)):
                segment_x.extend(self.x[i])
                segment_y.extend(self.y[i])
                segment_yerr.extend(self.yerr[i])
                if self.segments[i].counts is not None:
                    counts.extend(numpy.array(self.segments[i].counts, dtype=numpy.int_))
                else:
                    counts.extend(numpy.array(self.x[i])*numpy.nan) #numpy.zeros(self.x[i].size))

            segment_arrays = [numpy.array(segment_x),
                              numpy.array(segment_y),
                              numpy.array(segment_yerr),
                              counts]

            if all(numpy.isnan(count) for count in segment_arrays[3]):
                ascii.write(segment_arrays[0:3], filename, names=['x','y','y_err'], comment='#')
            else:
                ascii.write(segment_arrays, filename, names=['x','y','y_err','counts'], comment='#')


class AggregateSed(Stack):

    """ An Aggregate SED in the sense of Iris. Each Sed in Iris is an
    aggregation of different Segments representing the same astrophysical
    object. This class lets users differentiate between Segments, but shifts
    and normalizes the Segments as one, joined Segment.

    The redshift of each segment is not considered. Only the redshift of the
    AggregateSed object is used for red/blue-shifting (i.e., if aggsed is an AggregateSed
    object, use aggsed.z to access the redshift of the Segments inside aggsed.

    Notes
    -----
    ISSUES:
    - all methods turn Segments into Sed objects. Need to make sure
    Spectra stay as Spectra.
    - Need to deal with units at some point

    """

    def __init__(self, segments, z=None):
        Stack.__init__(self, segments)
        self.norm_constant = 1.0
        self.z = z

    @property
    def z(self):
        return self._z
    @z.setter
    def z(self, val):
        if isinstance(val, types.NoneType):
            self._z = val
        elif type(val) not in NUMERIC_TYPES:
            raise InvalidRedshiftError(0)
        elif val < 0:
            raise InvalidRedshiftError(1)
        else:
            self._z = numpy.float_(val)
            #for segment in self:
            #    segment.z = numpy.float_(val)
    @z.deleter
    def z(self):
        logging.info('Setting z to None.')
        self._z = None


    def _flatten(self):
        x = numpy.array([val for subl in self.x for val in subl])
        y = numpy.array([val for subl in self.y for val in subl])
        yerr = numpy.array([val for subl in self.yerr for val in subl])

        return x, y, yerr


    def _sorts(self):
        x, y, yerr = self._flatten()
        points = []
        for i, point in enumerate(x):
            points.append([x[i], y[i], yerr[i]])
        points = zip(*sorted(points))
        x = numpy.array(points[0])
        y = numpy.array(points[1])
        yerr = numpy.array(points[2])
        
        return x, y, yerr


    def shift(self, z0, correct_flux=True):
        spec, flux, fluxerr = self._sorts()

        if correct_flux:
           z_tot_flux, z0_tot_flux = shift(spec, flux, self.z, z0, norms=True)
            
        shifted_aggsed = AggregateSed([], z=z0)
        for segment in self:
            x = (1 + z0) * segment.x / (1+self.z)
            yerr = segment.yerr
            if correct_flux:
                y = segment.y * z_tot_flux / z0_tot_flux
            else:
                y = segment.y

            seg = Sed(x=x, y=y, yerr=yerr, 
                      z=z0
                      )
            shifted_aggsed.add_segment(seg)

        shifted_aggsed.z = z0
            
        # keep attributes of old sed
        _get_setattr(shifted_aggsed, self)

        return shifted_aggsed


    def normalize_by_int(self, minWavelength='min', maxWavelength='max', correct_flux=False, z0=None):
        spec, flux, fluxerr = self._sorts()
        flux = numpy.ma.masked_invalid(flux)
        fluxerr = numpy.ma.masked_invalid(fluxerr)

        if minWavelength == 'min':
            minWavelength=spec.min()
        if maxWavelength == 'max':
            maxWavelength=spec.max()

        lowCut = numpy.greater_equal(spec, minWavelength)
        highCut = numpy.less_equal(spec, maxWavelength)
        totalCut = numpy.logical_and(lowCut, highCut)
        sedWavelengthSlice = spec[totalCut]
        
        if len(sedWavelengthSlice) < 2:
            raise SegmentError('AggregateSed must have at least 2 points between minWavelength and maxWavelength.')

        if correct_flux:
            fluxz = correct_flux_(spec, flux, self.z, z0)
            flux = fluxz

        sedFluxSlice = flux[totalCut]
        norm_constant = 1.0/numpy.trapz(abs(sedFluxSlice), sedWavelengthSlice)

        norm_aggsed = AggregateSed([], z=self.z)
        for segment in self:
            x = segment.x
            y = segment.y*norm_constant
            yerr = segment.yerr*norm_constant if fluxerr is not None else None

            seg = Sed(x=x, y=y, yerr=yerr, 
                      z=self.z
                      )
            norm_aggsed.add_segment(seg)

        norm_aggsed.norm_constant = norm_constant
        # keep attributes of old sed
        _get_setattr(norm_aggsed,self)

        return norm_aggsed


    def normalize_at_point(self, x0, y0, norm_operator=0, dx=None, correct_flux=False, z0=None):
        spec, flux, fluxerr = self._sorts()

        if correct_flux:
            fluxz = correct_flux_(spec, flux, self.z, z0)
            flux = fluxz  

        # User decides if they want to average the flux of points between
        # x0-dx and x0+dx (kwarg dx=NUMBER), or if they want to use the 
        # flux of the nearest neighbor (kwarg dx=None)
        if isinstance(dx, types.NoneType):
            interp_self = calc.fast_nearest_interp([x0], spec, flux)
            flux_at_x0 = numpy.float_(interp_self)
        else:
            x0_idx = find_nearest(spec, x0)
            idx = (x0_idx-dx, x0_idx+dx)

            if idx[0] < 0 and idx[1] >= spec.size:
                raise OutsideRangeError
            elif idx[0] < 0:
                idx[0] = min(spec)
            elif idx[1] >= spec.size:
                idx[1] = max(spec)
            # to print out ranges used in the normalization.
            # Has no effect on the ranges used.
            if (idx[0] < 0) or (idx[1] >= spec.size):
                high_lim = min(spec[idx+dx], max(spec))
                low_lim = max(spec[idx-dx], min(spec))
                logger.warning(' Spectrum does not cover full range used for '+
                               'determining normalization constant. Spectral '+
                               'range used: [{low}:{high}]'.format(low=repr(low_lim), high=repr(high_lim)))
#            spec_indices = find_range(spec, x0-dx, x0+dx+1)
#            if spec_indices == (-1, -1):
#                raise OutsideRangeError
#            elif spec_indices[0] == -1:
#                spec_indices[0] = min(spec)
#            elif spec_indices[1] == -1:
#                spec_indices[1] = max(spec)
            # to print out ranges used in the normalization.
            # Has no effect on the ranges used.
#            if (x0-dx < min(spec)) or (x0+dx > max(spec)):
#                high_lim = min((x0+dx, max(spec)))
#                low_lim = max((x0-dx, min(spec)))
#                logger.warning(' Spectrum does not cover full range used for '+
#                               'determining normalization constant. Spectral '+
#                               'range used: [{low}:{high}]'.format(low=repr(low_lim), high=repr(high_lim)))
            try:
                flux_at_x0 = numpy.mean(flux[idx[0]:idx[1]])
            except FloatingPointError:
                flux_at_x0 = flux[idx[0]]
            
        if norm_operator == 0:
            norm_constant = y0 / flux_at_x0
        elif norm_operator == 1:
            norm_constant = y0 - flux_at_x0
        else:
            raise ValueError('Unrecognized norm_operator. keyword '+
                             '\'norm_operator\' must be either 0 '+
                             '(for multiply) or 1 (for addition).')

        norm_aggsed = AggregateSed([], z=self.z)
        for segment in self:
            x = segment.x
            if norm_operator == 0:
                y = segment.y*norm_constant
                yerr = segment.yerr*norm_constant if fluxerr is not None else None
            else:
                y = segment.y + norm_constant
                yerr = segment.yerr

            seg = Sed(x=x, y=y, yerr=yerr,
                      z=self.z
                      )
            norm_aggsed.add_segment(seg)

        norm_aggsed.norm_constant = norm_constant
        # keep attributes of old sed
        _get_setattr(norm_aggsed,self)

        return norm_aggsed
        


def stack(aggrseds, binsize, statistic, fill='remove', smooth=False, smooth_binsize=10, logbin=False):
    '''Rebins the SEDs along the spectral axis into user-defined binsizes, then combines the fluxes in each bin together according to one of four statistics: average, weighted average, addition, or a user-defined function.

    Parameters
    ----------
    aggrseds : array-like of sedstacker.sed.Segment; sedstacker.sed.Stack
        The Segments, AggregateSeds and/or Stacks to combine. May be an iterable of Stacks.
    binsize : float or int
        numerical value to bin the spectral axis by. The fluxes within each bin are combined according to the statistic argument. binsize is also the resolution of the stacked SED.
    statistic : str or func
        The statistic to use for combining the fluxes in each bin. The possible statistics are:
            - *'avg'* (default) - averages the fluxes within each bin using numpy.average
            - *'wavg'* - computes the weighted average for the fluxes within each bin using the corresponding flux-errors and numpy.average. If a point has no error associated with it, then it is excluded from the calculation.
            - *'sum'* - sums the fluxes within each bin.
            - *func* - a user-defined function for combining the fluxes in each bin. It must accept three arguments: the first and second arguments as arrays for the flux and flux-error values in a bin, and the third argument is a boolean which is False if there are no NaN-valued flux errors in the list of SEDs to be stacked. The function must return the combined flux and flux-error values, and the number of flux values within the bin.

            Ex:

            >>> def my_weighted_avg(flux_bin, flux_err_bin, nans):
                    if nans:
                        # Removes points without flux errors from the bin
                        # therefore removing them from the calculations.
                        flux_err_bin = flux_err_bin[~numpy.isnan(flux_err_bin)]
                        flux_bin = flux_bin[~numpy.isnan(flux_err_bin)]
                        weights = 1.0/flux_error_bin**2
                        flux = numpy.ma.average(flux_bin, weights=weights)
                        flux_err = numpy.sqrt((flux_err_bin**2).sum())
                    else:
                        weights = 1.0/flux_error_bin**2
                        flux = numpy.ma.average(flux_bin, weights=weights)
                        flux_err = numpy.sqrt((flux_err_bin**2).sum())
                    counts = len(flux_err_bin)

                    return yarr, outerr, counts

    fill : str
        Switch that decides what to do with bins that have no flux counts in them. There are two options:
            - *'fill'* - the Y-values of unpopulated X-values are assigned numpy.nan
            - *'remove'* - (Default) removes the unpopulated X-Y pair from the stacked SED
    smooth : bool
        Specifies whether the stacked SED should be smoothed using a boxcar method (*True*) or not (*False*). Default is *False*.
    smooth_binsize : int
        The size of the boxcar. Default is 10.
    logbin : bool
        Specifies how to bin the Stack
            - *False* - (default) linear binning
            - *True* - logarithmic binning

    Returns
    -------
    sedstacker.sed.Sed
        SED with attribute 'count'. Attributes z, xunit and yunit are taken from the first Segment or AggregateSed in the list.
            counts - number of flux values combined per binsize.
            stack() calculates the error of the combined fluxes for each bin. If all points in the input SEDs have flux errors associated to them, then ``stack()`` returns the square of the sum of the errors in quadrature as the flux error. Otherwise, the returned Sed's flux error is the standard deviation of flux values in each bin.

    Examples
    --------
    Stack a group of spectra with binsize 1.0 and average statistic

    >>> stacked_spectra = stack(spectra, 1.0, 'avg')

    Stack a group of spectra with binsize 1.0, a user-defined statistic, then smooth the stacked spectrum. The user-defined statistic function must accept and return the combined flux, flux-error, and the number of flux values that fall within the bin (counts) in this order

    >>> def my_combination_func(flux, flux_error, counts):
            flux_out = numpy.average(flux)
            flux_error_out = numpy.average(flux_error)
            return flux_out, flux_error_out, counts
    >>> 
    >>> smooth_stacked_spectra = stack(spectra, 1.0, my_combination_func, smooth=True, smooth_binsize=5)

    Stack a group of SEDs with logarithmic binning and weighted average statistic

    >>> stacked_seds = stack (seds, 0.1, 'wavg', logbin=True)

    Notes
    -----
    - If logbin=True, the binsize should be entered in logspace. E.g. if 1,000 < x < 100,000 (or 3.0 < log10(x) < 5.0), and you wish to bin the spectral axis evenly in logspace, choosing binsize = 0.1 will produce 20 bins.
    - If the stack's spectral axis ranges over 3 or 4 decades, it is advised to set logbbin = True
    - Assuming ``stacked_sed`` is the return value of ``stack()``, plotting ``stacked_sed.counts`` against ``stacked_sed.x`` gives a histogram of the flux counts per spectral bin.
    - If there are any points without flux errors, then the ``yerr`` attribute of the stacked SED will be the variance of the fluxes in each bin.

    '''

    if type(binsize) not in NUMERIC_TYPES:
        raise ValueError('binsize must be of numeric type int or float.')

    if type(smooth) not in (types.BooleanType, numpy.bool_):
        raise ValueError('keyword argument smooth must be \'True\' or \'False\'.')

# Haven't implemented units yet, so we only care about the first element in binsize.
#    if len(binsize) != 2:
#        raise Exception('Argument binsize must be a tuple:'+
#                        '(binsize, \'units\')')
#    if not isinstance(binsize[1], str):
#        raise Exception('units must be a string recognized by the AstroPy.units package')

    # making "giant"/global arrays
    giant_spec = numpy.array([])
    giant_flux = numpy.array([])
    giant_fluxerr = numpy.array([])

    for i, sed in enumerate(aggrseds):
        giant_spec = numpy.append(giant_spec, sed.x)
        giant_flux = numpy.append(giant_flux, sed.y)
        giant_fluxerr = numpy.append(giant_fluxerr, sed.yerr)

    xarr = calc.big_spec(giant_spec, binsize, logbin)
    start = time.clock()
    yarr, xarr, yerrarr, counts = calc.binup(giant_flux, giant_spec, xarr,
                                             statistic, binsize, fill,
                                             giant_fluxerr, logbin=logbin)
    end = time.clock()
    # take first entry of xunit, yunit and z from the aggregate SED
    # for the same attributes in the stacked SED
    xunit = [aggrseds[0].xunit[0]]*xarr.size
    yunit = [aggrseds[0].yunit[0]]*xarr.size
    try:
        z = aggrseds[0][0].z
    except TypeError:
        z = aggrseds[0].z
    except AttributeError:
        z = aggrseds[0].z

    # for smoothing.
    # what to do with y-errors?
    if smooth:
        yarr = calc.smooth(yarr, smooth_binsize)

    stacked_sed = Sed(x=xarr, y=yarr, yerr=yerrarr,
                      xunit=xunit, yunit=yunit, z=z)

    stacked_sed.counts = counts

    return stacked_sed


def create_from_points(points, z=None):
    '''Creates and returns a Sed object from an iterable of PhotometricPoints.

    Parameters
    ----------
    points : array-like 
        A collection of sedstacker.sed.PhotometricPoint's to populate a Sed
    z : float
        The redshift of the Sed. Default is None.

    Examples
    --------

    >>> # Create list of dummy PhotometricPoints
    >>> 
    >>> points = [PhotometricPoint(x=1+i, y=1+i, yerr=0.1*i) for i in range(10)]
    >>>
    >>> # Create a Sed from 'points'
    >>> 
    >>> sed = create_from_points(points)
    >>> print sed
      x   y  yerr xunit     yunit     
     --- --- ---- ----- --------------
      1   1  0.0    AA erg/s/cm**2/AA
      2   2  0.1    AA erg/s/cm**2/AA
      3   3  0.2    AA erg/s/cm**2/AA
      4   4  0.3    AA erg/s/cm**2/AA
      5   5  0.4    AA erg/s/cm**2/AA
      6   6  0.5    AA erg/s/cm**2/AA
      7   7  0.6    AA erg/s/cm**2/AA
      8   8  0.7    AA erg/s/cm**2/AA
      9   9  0.8    AA erg/s/cm**2/AA
     10  10  0.9    AA erg/s/cm**2/AA

    '''
    
    sed = Sed()
    for point in points:
        sed.add_point(point)
    sed.z = z
    return sed


def shift(spec, flux, z, z0, norms=False):
    '''Redshift spectral/SED data.

    Parameters
    ----------
    spec : array-like
        Spectral coordinates at observed redshift, z
    flux : array-like
        Flux values at observed redshift, z
    z : float or int
        Observed redshift of the SED/spectrum
    z0 : float or int
        Target redshift to shift the SED/spectrum to
    norms : bool
        If True, shift returns a tuple of the integrated flux of the SED at 
        redshift z and and redshift z0.

    Returns
    -------
    spec_z0 : array-like
        Shifted spectral coordinates
    flux_z0 : array-like
        Flux corrected for the intrinsic dimming/brightening due to shifting the SED

    '''

    if z is None:
        raise NoRedshiftError
    if type(z0) not in NUMERIC_TYPES:
        raise InvalidRedshiftError(0)
    if type(z) not in NUMERIC_TYPES:
        raise InvalidRedshiftError(0)
    if z0 < 0:
        raise InvalidRedshiftError(1)

    spec = numpy.ma.masked_invalid(spec)
    flux = numpy.ma.masked_invalid(flux)

    spec_z0 = (1 + z0) * spec / (1+z)
    
    z0_total_flux = numpy.trapz(spec_z0, flux)
    z_total_flux = numpy.trapz(spec, flux)
    
    flux_z0 = flux*z_total_flux/z0_total_flux
    
    if norms == False:
        return numpy.array(spec_z0), numpy.array(flux_z0)
    else:
        return z_total_flux, z0_total_flux


def correct_flux_(spec, flux, z, z0):
    specz0 = spec * (1+z0) / (1+z)
    tmp, corrected_flux = shift(specz0, flux, z0, z)
    return corrected_flux


def find_range(array, a, b):

    '''Find elements in array that are greater than a and less than b.
    Returns a tuple of the array indices (start, end), such that array[start]
    is the first value and array[end] is the last value.
    If no value is found, it returns (-1, -1).'''

    start = bisect_right(array, a)
    end = bisect_left(array, b)
    if start == end:
        return (-1, -1)
    else:
        return [start-1, end]

def find_nearest(a, a0):
    return numpy.abs(a - a0).argmin()


def _get_setattr(new_object, old_object):
    for key in old_object.__dict__:
        if key not in new_object.__dict__:
            setattr(new_object, key, old_object.__dict__[key])

