import copy
import logging
import numpy
import os.path
import types
from math import log10
from bisect import bisect_left, bisect_right

from astropy.io import ascii
from astropy.table import Table
from scipy import interpolate, integrate

from sedstacker import calc
from sedstacker.exceptions import NoRedshiftError, InvalidRedshiftError, SegmentError, OutsideRangeError, NotASegmentError, PreExistingFileError

# UNRESOLVED ISSUES
#
# 1. shifting SEDs - right now, only works for x in wavelength units.
#                    need to implement for other units, like freq and
#                    energy


logging.basicConfig(format='%(levelname)s:%(message)s')


class PhotometricPoint(object):
    '''Represents a photometric point on a SED.
    Attributes: x, y, yerr, xunit, yunit, units'''

    def __init__(self, x=None, y=None, yerr=None, xunit='AA', yunit='erg/s/cm**2/AA'):
        '''Returns a PhotometricPoint.
        Attributes: x, y, yerr, xunit, yunit

        >>> point = PhotometricPoint(x=3823.0, y=1e-16, yerr=1e-18)
        >>> print point.x
        3823.0'''
                
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
        '''Prints a PhotometricPoint object.

        >>> point = PhotometricPoint(x=3823.0, y=1e-16, yerr=1e-18)
        >>> print point
        (3823 erg/s/cm**2/AA, 1e-16 +/- 1e-18 erg/s/cm**2/AA)
        >>> point2 = PhotometricPoint(x=3823.0, y=1e-16)
        >>> print point2
        (3823 erg/s/cm**2/AA, 1e-16 erg/s/cm**2/AA)'''

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

    def __init__(self, x=[], y=[], yerr=None, xunit='AA', yunit='erg/s/cm**2/AA', z=None):
        '''Creates and Returns a Spectrum.
        Kwargs:
            x (list): The spectral coordinates. Default value is None.
            y (list): The flux values. Default value is None.
            yerr (list, float, int): The errors on the flux values. Default value is None.
            xunit (str): The spectral coordinate units. Default value is 'AA'.
            yunit (str): The flux coordinates. Default value is 'erg/s/cm**2/AA'.
            z (float): The redshift of the Sed. Default value is None.'''

        yerrtypes = (types.FloatType, numpy.float_, types.IntType, numpy.int_)

        if len(x) != len(y):
            raise SegmentError('x and y must be of the same length.')

        self.x = numpy.array(x)
        self.y = numpy.array(y)

        if yerr is None:
            self.yerr = numpy.array([numpy.nan]*len(y))
        elif type(yerr) in yerrtypes:
            self.yerr = numpy.array([float(yerr)]*len(y))
        elif len(yerr) == len(y):
            self.yerr = numpy.array(yerr)
        else:
            raise SegmentError('y and yerr must be of the same length.')

        self.xunit = xunit
        self.yunit = yunit

        # How do you assert that an attribute must be of a certain type?
        # Say someone adds a redshift after creating the Spectrum object. How can I check that the attribute value they enter is non-negative and of numeric type?
        if isinstance(z, types.NoneType):
            self.z = z
        elif type(z) not in (types.FloatType, types.IntType, numpy.float_, numpy.int_):
            raise InvalidRedshiftError(0)
        elif z < 0:
            raise InvalidRedshiftError(1)
        else:
            self.z = z


    def shift(self, z0, correct_flux=True):
        '''
        Redshifts the spectrum by means of cosmological expansion.
        
        Args:
            z0 (float, int): Target redshift to shift the SED/spectrum to
        
        Kwargs:
            correct_flux (bool): If True, the flux will be corrected for the
            intrinsic dimming/brightening due to shifting the spectrum.
            If False, only the spectral coordinates will be shifted; the flux
            remains the same.

        Returns:
            A new Spectrum object with the redshifted spectrum.
        '''

        spec = self.x
        flux = self.y

        if correct_flux:
            spec_z0, flux_z0 = shift(spec, flux, self.z, z0)
        else:
            spec_z0 = (1 + z0) * spec / (1+self.z)
            flux_z0 = flux

        spec = Spectrum(x=spec_z0, y=flux_z0, yerr=self.yerr,
                        xunit=self.xunit, yunit=self.yunit, z=z0)
        # keep attributes of old sed
        _get_setattr(spec,self)

        return spec


    def normalize_at_point(self, x0, y0, dx=50, norm_operator=0, correct_flux=False, z0=None):

        '''Normalizes the SED such that at spectral coordinate x0,
        the flux of the SED is y0.

        normalize_at_point() takes the average flux within a range of spectral values [x0-dx, x0+dx] centered on x0 as the observed flux at x0.

        >>> # initializing dummy spectrum
        >>> x = numpy.arange(3000,9500,0.5)
        >>> y = numpy.random.rand(x.size)
        >>> yerr = y*0.01
        >>> spec = Spectrum(x=x,y=y,yerr=yerr,z=0.3)
        >>>
        >>> # normalize the spectrum "spec" to 1.0 erg/s/cm**2/AA at 3600.0 AA
        >>> norm_spec = spec.normalize_at_point(3600.0, 1.0, dx=70)
        >>> norm_spec.y
        array([ 1.13493095,  0.40622771,  0.61081693, ...,  1.87997373,
                0.48542037,  0.02411532])
        >>>
        >>> norm_spec.norm_constant
        2.0265575749283653

        Args:
            x0 (float, int): The spectral coordinate to normalize the SED at. x0 is in Angstroms.
            y0 (float, int): The flux value to normalize the SED to.

        Kwargs:
            dx (float, int): The number of spectral points to the left and right of x0, over which the average flux is measured. If no points fall within the range [x0-dx,x=+dx], then OutsideRangeError is raised, and the normalization is aborted.
            norm_operator (int): operator used for scaling the spectrum to y0.
                0 = multiply the flux by the normalization constant
                1 = add the normalization constant to the flux
            correct_flux (bool): kwarg to correct for flux dimming/brightening due to redshift. Meant for SEDs that were shifted only by wavelength (i.e. the flux was not corrected for the intrinsic dimming/brightening due to redshift). If correct_flux = True, then the flux is corrected so that the integrated flux at the current redshift is equal to that at the original redshift.Default value is False.
            z0 (float or int): The original redshift of the source. Used only if correct_flux = True.

        Requires that the SED has at least 4 photometric points'''

        numpy.seterr(invalid='raise')

        if len(self.x) < 4:
            raise SegmentError('Spectrum object must have 4 or more points to use normalize_at_point().')

        y0 = numpy.float_(y0)
        x0 = numpy.float_(x0)
        dx = numpy.float_(dx)
        flux = numpy.ma.masked_invalid(self.y)
        fluxerr = self.yerr

        spec_indices = find_range(self.x, x0-dx, x0+dx+1)
        if spec_indices == (-1, -1):
            raise OutsideRangeError
        elif spec_indices[0] == -1:
            spec_indices[0] = min(self.x)
        elif spec_indices[1] == -1:
            spec_indices[1] = max(self.x)
        if (x0-dx < min(self.x)) or (x0+dx > max(self.x)):
            high_lim = min((x0+dx, max(self.x)))
            low_lim = max((x0+dx, min(self.x)))
            logging.warning(' Spectrum does not cover full range used for determining normalization constant. Spectral range used: [{low}:{high}]'.format(low=repr(low_lim), high=repr(high_lim)))

        if correct_flux:
            fluxz = correct_flux_(self.x, flux, self.z, z0)
            flux = fluxz

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

        Kwargs:
            minWavelength (float or 'min'): minimum wavelength of range over which to normalize Spectrum
            maxWavelength (float or 'max'): maximum wavelength of range over which to normalize Spectrum
            correct_flux (bool): switch used to correct for SEDs that were shifted to some redshift without taking into account flux brightening/dimming due to redshift
            z0 (float or int): The original redshift of the source. Used if correct_flux = True.

        The Spectrum must have at least 2 points between minWavelength and maxWavelength.
        '''

        flux = numpy.ma.masked_invalid(self.y)
        fluxerr = self.yerr

        if minWavelength == 'min':
            minWavelength=self.x.min()
        if maxWavelength == 'max':
            maxWavelength=self.x.max()

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


    def write(self, filename, xunit='AA', yunit='erg/s/cm**2/AA', fmt='ascii'):
        '''Write Spectrum to file.
        Ex:
        # x y
        1941.8629     0.046853197
        1942.5043     0.059397754
        1943.1456     0.032893488
        1943.7870     0.058623008
        ...            ... 
        10567.7890     0.046843890
        10568.4571     0.059888754

        Args:
            filename (str): The name of the output file.
       
        Kwargs:
            xunit (str): Default unit is Angstroms. Converts all the spectral data in Spectrum to these units.
            yunit (str): Default unit is erg/s/cm**2/AA. Converts all the flux data in Spectrum to these units.
            fmt (str): The format for the output file. The default file format is 'ascii'. For release 1.0, only ASCII files will be supported.'''

        if os.path.exists(filename):
            raise PreExistingFileError(filename)
        else:
            segment_arrays = [self.x, self.y, self.yerr]
            ascii.write(segment_arrays, filename, names=['x','y','y_err'], comment='#')


class Sed(Segment, list):
    '''Represents a photometric SED from one astronomical object or model.'''

    def __init__(self, x=[], y=[], yerr=None, xunit=['AA'], yunit=['erg/s/cm**2/AA'], z=None):
        '''
        Kwargs:
            x (list): The spectral coordinates. Default value is an empty list, [].
            y (list): The flux values. Default value is an empty list, [].
            yerr (list; float, int): The errors on the flux values. Default value is None. If all y-values share the same error, yerr can be a float or integer.
            xunit (list, str): The spectral coordinate units. Default value is ['AA'].
            yunit (list, str): The flux coordinates. Default value is ['erg/s/cm**2/AA'].
            z (float): The redshift of the Sed. Default value is None.
        Raises:
            SegmentError
        
        x and y must have the same length. If the kwarg of yerr is a single value YERR, then all (x,y) SED points will have error YERR.
        >>> sed = Sed(x=[1212.0, 3675.0, 4856.0], y=[1.456e-11, 3.490e-11, 5.421e-11], yerr=1.0e-13, z=0.02)
        >>> for point in sed:
        ...     print point.yerr
        ...
        1.0e-13
        1.0e-13
        1.0e-13

'''


#        self._cache = []
        if isinstance(z, types.NoneType):
            self.z = z
        elif type(z) not in (types.FloatType, types.IntType, numpy.float_, numpy.int_):
            raise InvalidRedshiftError(0)
        elif z < 0:
            raise InvalidRedshiftError(1)
        else:
            self.z = z
        
        if len(x) != len(y):
            raise SegmentError('x and y must be of the same length.')

        if yerr is None:
            yerr = [numpy.nan]*len(x)
        else:
            if len(yerr) == 1:
                yerr = [yerr]*len(x)
            elif len(yerr) != len(x):
                raise SegmentError('x and yerr must be of the same length.')

        if len(xunit) == 1:
            xunit = xunit*len(x)
        elif len(xunit) != len(x):
            raise SegmentError('xunit must have the same length as x.')

        if len(yunit) == 1:
            yunit = yunit*len(y)
        elif len(yunit) != len(y):
            raise SegmentError('yunit must have the same length as y.')
        
        for i in range(len(x)):
            point = PhotometricPoint(x=x[i], y=y[i], yerr=yerr[i], xunit=xunit[i], yunit=yunit[i])
            self.append(point)


#    def set_cache(self):        
#        self._cache = numpy.array(x, y, yerr, xunit, yunit)
#    def update_cache(self):


    def shift(self, z0, correct_flux = True):
        '''
        Redshifts the SED by means of cosmological expansion.

        Args:
            z0 (float, int): Target redshift to shift the SED/spectrum to
        
        Kwargs:
            correct_flux (bool): If True, the flux will be corrected for the intrinsic
            dimming/brightening due to shifting the spectrum.
            If False, only the spectral coordinates will be shifted; the flux remains
            the same.
        
        Returns:
            A new Sed object with the redshifted SED.
        '''

        sedarray = self.toarray()
        spec = sedarray[0]
        flux = sedarray[1]
        fluxerr = sedarray[2]
        xunit = sedarray[3]
        yunit = sedarray[4]

        if correct_flux:
            spec_z0, flux_z0 = shift(spec, flux, self.z, z0)
        else:
            spec_z0 = (1 + z0) * spec / (1+self.z)
            flux_z0 = flux
        sed = Sed(x=spec_z0, y=flux_z0, yerr=fluxerr,
                   xunit=xunit, yunit=yunit, z=z0)

        # keep attributes of old sed
        _get_setattr(sed,self)

        return sed


    def normalize_at_point(self, x0, y0, norm_operator=0, correct_flux=False, z0=None):

        '''Normalizes the SED such that at spectral coordinate x0,
           the flux of the SED is y0.

        Uses nearest-neighbor interpolation.

        Args:
            x0 (float, int): The spectral coordinate to normalize the SED at
            y0 (float, int): The flux value to normalize the SED to

        Kwargs:
            correct_flux (bool): kwarg to correct for flux dimming/brightening
            due to redshift. Meant for SEDs that were shifted only by wavelength
            (i.e. the flux was not corrected for the intrinsic
            dimming/brightening due to redshift). If correct_flux = True, then
            the flux is corrected so that the integrated flux at the current
            redshift is equal to that at the original redshift.Default value is False.

            norm_operator (int): operator used for scaling the spectrum to y0.
            0 = multiply the flux by the normalization constant
            1 = add the normalization constant to the flux

            z0 (float or int): The original redshift of the source.
            Used only if correct_flux = True.

            '''

        sedarray = self.toarray()
        spec = sedarray[0]
        flux = sedarray[1]
        fluxerr = sedarray[2]
        xunit = sedarray[3]
        yunit = sedarray[4]

        if correct_flux:
            fluxz = correct_flux_(spec, flux, self.z, z0)
            flux = fluxz  

        interp_self = interpolate.interp1d(spec, flux, kind='nearest')
        flux_at_x0 = interp_self(x0)

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
        setattr(norm_sed, 'norm_constant', norm_constant)

        # keep attributes of old sed
        _get_setattr(norm_sed,self)

        return norm_sed
    

    def normalize_by_int(self, minWavelength='min', maxWavelength='max', correct_flux=False, z0=None):

        '''Normalises the SED such that the area under the specified wavelength range is equal to 1.

        Algorithm adopted from astLib.astSED.normalise(); uses the Trapezoidal rule to estimate the integrated flux.

        Kwargs:
            minWavelength (float or 'min'): minimum wavelength of range over which to normalise SED
            maxWavelength (float or 'max'): maximum wavelength of range over which to normalise SED
            correct_flux (bool): switch used to correct for SEDs that were shifted to some redshift without taking into account flux brightening/dimming due to redshift
            z0 (float or int): The original redshift of the source. Used if correct_flux = True.

        The SED must have at least 2 points between minWavelength and maxWavelength.

        '''

        sedarray = self.toarray()
        spec = sedarray[0]
        flux = numpy.ma.masked_invalid(sedarray[1])
        fluxerr = sedarray[2]
        xunit = sedarray[3]
        yunit = sedarray[4]

        if minWavelength == 'min':
            minWavelength=spec.min()
        if maxWavelength == 'max':
            maxWavelength=spec.max()

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
        norm_constant = 1.0/integrate.trapz(abs(sedFluxSlice), sedWavelengthSlice)
        norm_flux = numpy.array(flux*norm_constant)
        norm_fluxerr = numpy.array(fluxerr*norm_constant if fluxerr is not None else None)
        norm_segment = Sed(x=spec, y=norm_flux, yerr=norm_fluxerr,
                           xunit=xunit, yunit=yunit, z=self.z)

        setattr(norm_segment, 'norm_constant', norm_constant)
        # keep attributes of old sed
        _get_setattr(norm_segment,self)

        return norm_segment


    def add_point(self, point):
        '''Add a PhotometricPoint to Sed object.
        Args:
            point (PhotometricPoint): A PhotometricPoint object.'''
        self.append(point)


    def remove_point(self, index):
        '''Remove a PhotometricPoint from Sed object.
        Args:
            index (int): the index of the PhotometricPoint in the list to remove.'''
        self.pop(index)


    def mask_point(self, index):
        '''Mask a point. 
        Args:
            index (int): the index of the PhotometricPoint in the list to mask.'''
        
        self[index].mask = True


    def unmask_point(self, index):
        '''Unmask a point. 
        Args:
            index (int): the index of the PhotometricPoint in the list to unmask.'''
        
        self[index].mask = False


    def add_segment(self, *segments):
        '''Add a segment or iterable of segments to the Sed. '''
        
        for segment in segments:
            for points in segment:
                self.append(points)


    def toarray(self):
        '''Convert a Sed to a 5-dimensional array.

        Example: If

        >>> sedarray = Sed().toarray()

        then

        sedarray[0] --> spectral axis, 'x'
        sedarray[1] --> flux axis, 'y'
        sedarray[2] --> flux-error axis, 'yerr'
        sedarray[3] --> xunit values, 'xunit'
        sedarray[4] --> yunit values, 'yunit'
        
        Returns:
            5-dimensional array containing the spectral, flux, flux-error,
            xunit and yunit values in the Sed.
        '''
        
        spec = numpy.array([p.x for p in self])
        flux = numpy.array([p.y for p in self])
        fluxerr = numpy.array([p.yerr for p in self])
        xunit = numpy.array([p.xunit for p in self])
        yunit = numpy.array([p.yunit for p in self])

        #sedarray = Table({'x':spec,
        #                 'y':flux,
        #                 'yerr':fluxerr,
        #                 'xunit':xunit,
        #                 'yunit':yunit},
        #                 names=['x','y','yerr','xunit','yunit'])

        sedarray = spec, flux, fluxerr, xunit, yunit

        return sedarray


    def write(self, filename, xunit='AA', yunit='erg/s/cm**2/AA', fmt='ascii'):
        '''Write Sed to file.
        Ex:

        x y
        1941.8629     0.046853197
        1942.5043     0.059397754
        1943.1456     0.032893488
        1943.7870     0.058623008
        ...            ... 
        10567.7890     0.046843890
        10568.4571     0.059888754

        Args:
            filename (str): The name of the output file.
       
        Kwargs:
            xunit (str): Default unit is Angstroms. Converts all the spectral data in Sed to these units.
            yunit (str): Default unit is erg/s/cm**2/AA. Converts all the flux data in Sed to these units.
            fmt (str): The format for the output file. The default file format is 'ascii'. For release 1.0, only ASCII files will be supported.'''

        if os.path.exists(filename):
            raise PreExistingFileError(filename)
        else:
            sed = self.toarray()
            if hasattr(self, 'counts'):
                segment_arrays = Table({'x':sed[0],
                                        'y':sed[1],
                                        'y_err':sed[2],
                                        'counts':self.counts},
                                       names=['x','y','y_err','counts'],
                                       dtypes=('f10','f10','f10','i5'))
                ascii.write(segment_arrays, filename, comment='#', names=['x','y','y_err', 'counts'])
            else:
                segment_arrays = [sed[0],sed[1],sed[2]]
                ascii.write(segment_arrays, filename, names=['x','y','y_err'], comment='#')


class AggregateSed(list):

    def __init__(self, segments):

        self.segments = segments

        self.x = []
        self.y = []
        self.yerr = []
        self.xunit = []
        self.yunit = []
        self.z = []

        for segment in segments:
            if not isinstance(segment, Segment):
                raise NotASegmentError
            elif isinstance(segment, Sed):
                sedarray = segment.toarray()
                self.x.append(sedarray[0])
                self.y.append(sedarray[1])
                self.yerr.append(sedarray[2])
                self.xunit.append(sedarray[3])
                self.yunit.append(sedarray[4])
                self.z.append(segment.z)
            else:
                self.x.append(segment.x)
                self.y.append(segment.y)
                self.yerr.append(segment.yerr)
                self.xunit.append(segment.xunit)
                self.yunit.append(segment.yunit)
                self.z.append(segment.z)
            self.append(segment)

#        self = self
#        self.x = self.x
#        self.y = self.y
#        self.yerr = self.yerr
#        self.xunit = self.xunit
#        self.yunit = self.yunit
#        self.z = self.z

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
        Redshifts the Seds and/or Spectra in the AggregateSed by means of cosmological
        expansion.

        Args:
            z0 (float, int): Target redshift to shift the AggregateSed to
        
        Kwargs:
            correct_flux (bool): If True, the flux will be corrected for the intrinsic
            dimming/brightening due to shifting the spectrum.
            If False, only the spectral coordinates will be shifted; the flux remains
            the same.

        Raises:
            NoRedshift: Raised if a Sed/Spectrum has no redshift. If raised, the segment
            is excluded from the returned shifted AggregateSed.
            InvalidRedshiftError: Raised if z0 is not of numeric type Float or Integer, or if z0 is negative. If raised, the segment is excluded from the returned shifted AggregateSed.
        
        Returns:
            A new AggregateSed object with the redshifted SED.
        '''

        shifted_segments = []

        for segment in self.segments:
            try:
                shifted_seg = segment.shift(z0, correct_flux = correct_flux)
                shifted_segments.append(shifted_seg)
            except NoRedshiftError:
                logging.warning(' Excluding AggregateSed[%d] from the shifted AggregateSed.' % self.index(segment))
                pass

            except InvalidRedshiftError:
                logging.warning(' Excluding AggregateSed[%d] from the shifted AggregateSed.' % self.index(segment))
                pass

        return AggregateSed(shifted_segments)


    def filter(self, boolean = '>', **kwargs):
        raise NotImplemented('filter() is not implemented yet.')


    def normalize_at_point(self, x0, y0, norm_operator=0, correct_flux=False, z0=None):
        '''Normalizes the SED such that at spectral coordinate x0,
           the flux of the SED is y0.

        Args:
            x0 (float, int): The spectral coordinate to normalize the SED at
            y0 (float, int): The flux value to normalize the SED to

        Kwargs:
            correct_flux (bool): kwarg to correct for flux dimming/brightening due to redshift. Meant for SEDs that were shifted only by wavelength (i.e. the flux was not corrected for the intrinsic dimming/brightening due to redshift). If correct_flux = True, then the flux is corrected so that the integrated flux at the current redshift is equal to that at the original redshift.Default value is False.
            norm_operator (int): operator used for scaling the spectrum to y0.
            0 = multiply the flux by the normalization constant
            1 = add the normalization constant to the flux
            z0 (float or int): A list or array of the original redshifts of the sources. The redshifts must be in the same order as the Segments appear in the AggregateSed (i.e. assuming aggsed is an AggregateSed, z0[0] is the original redshift of aggsed[0], z0[1] is the original redshift of aggsed[1], etc.). Used only if correct_flux = True.

        Raises:
            OutisdeRange: If a Sed's or Spectrum's spectral range does not cover point x0,
            then the segment is excluded from the returned normalized AggregateSed.
            SegmentError: If a Sed or Spectrum has less than 4 points, then the Sed or
            Spectrum is excluded from the returned normalized AggregateSed.

        Requires that the Seds has at least 4 photometric points'''

        if isinstance(z0,(types.FloatType, types.IntType, numpy.float_,numpy.int_,types.NoneType)):
            z0 = [z0]*len(self.segments)
        elif len(z0) != len(self.segments):
            raise ValueError('Length of z0 does not match the length of AggregateSed.')

        norm_segments = []

        for i, segment in enumerate(self.segments):
            try:
                norm_seg = segment.normalize_at_point(x0, y0,
                                                      norm_operator=norm_operator,
                                                      correct_flux=correct_flux,
                                                      z0=z0[i])
                norm_segments.append(norm_seg)
            except OutsideRangeError:
                logging.warning(' Excluding AgggregateSed[%d] from the normalized AggregateSed.' % self.index(segment))
                pass
            except SegmentError, e:
                logging.warning(' Excluding AggregateSed[%d] from the normalized AggregateSed' % self.index(segment))
                pass

        return AggregateSed(norm_segments)


    def normalize_by_int(self, minWavelength='min', maxWavelength='max', correct_flux=False, z0=None):
        '''Normalises the SED such that the area under the specified wavelength range is equal to 1.

        Kwargs:
            minWavelength (float or 'min'): minimum wavelength of range over which to normalise SED
            maxWavelength (float or 'max'): maximum wavelength of range over which to normalise SED
            correct_flux (bool): switch used to correct for SEDs that were shifted to some redshift without taking into account flux brightening/dimming due to redshift
            z0 (float or int): A list or array of the original redshifts of the sources. The redshifts must be in the same order as the Segments appear in the AggregateSed (i.e. assuming aggsed is an AggregateSed, z0[0] is the original redshift of aggsed[0], z0[1] is the original redshift of aggsed[1], etc.). Used only if correct_flux = True.
        Raises:
            OutisdeRangeError: If a Sed's or Spectrum's spectral range does not cover point x0,
            then the segment is excluded from the returned normalized AggregateSed.
            SegmentError: If a Sed or Spectrum has less than 4 points, then the Sed or
            Spectrum is excluded from the returned normalized AggregateSed.

        Requires that the Segments have at least 4 points.

        '''

        if isinstance(z0,(types.FloatType, types.IntType, numpy.float_,numpy.int_,types.NoneType)):
            z0 = [z0]*len(self.segments)
        elif len(z0) != len(self.segments):
            raise ValueError('Length of z0 does not match length of AggregateSed.')

        norm_segments = []

        for i, segment in enumerate(self.segments):
            try:
                norm_seg = segment.normalize_by_int(minWavelength=minWavelength,
                                                    maxWavelength=maxWavelength,
                                                    correct_flux=correct_flux,
                                                    z0 = z0[i])
            except SegmentError:
                logging.warning(' Excluding AggregateSed[%d] from the normalized AggregateSed' % self.index(segment))
                pass

            norm_segments.append(norm_seg)

        return AggregateSed(norm_segments)


    def add_segment(self, segment):
        '''Add a segment to the AggregateSed.
           Args:
               segment: A Segment object.'''

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

            self.z.append(segment.z)
            if isinstance(segment, Spectrum):
                self.x.append(segment.x)
                self.y.append(segment.y)
                self.yerr.append(segment.yerr)
                self.xunit.append(segment.xunit)
                self.yunit.append(segment.yunit)
            else:
                sedarray = segment.toarray()
                self.x.append(sedarray[0])
                self.y.append(sedarray[1])
                self.yerr.append(sedarray[2])
                self.xunit.append(sedarray[3])
                self.yunit.append(sedarray[4])
        else:
            raise NotASegmentError


    def remove_segment(self, segment):
        '''Remove a segment from the AggregateSed.
           Args:
               segment: A Segment object.'''

        index = self.index(segment)
        self.x.pop(index)
        self.y.pop(index)
        self.yerr.pop(index)
        self.xunit.pop(index)
        self.yunit.pop(index)
        self.z.pop(index)

        self.remove(segment)
        self.segments.remove(segment)


    def write(self, filename, xunit='AA', yunit='erg/s/cm**2/AA', fmt='ascii'):
        '''Write Sed to file.
        Ex:
        # x y
        1941.8629     0.046853197
        1942.5043     0.059397754
        1943.1456     0.032893488
        1943.7870     0.058623008
        ...            ... 
        10567.7890     0.046843890
        10568.4571     0.059888754

        Args:
            filename (str): The name of the output file.
       
        Kwargs:
            xunit (str): Default unit is Angstroms. Converts all the spectral data in Sed to these units.
            yunit (str): Default unit is erg/s/cm**2/AA. Converts all the flux data in Sed to these units.
            fmt (str): The format for the output file. The default file format is 'ascii'. For release 1.0, only ASCII files will be supported.'''
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
                if hasattr(self.segments[i], 'counts'):
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


def stack(aggrseds, binsize, statistic, fill='remove', smooth=False, smooth_binsize=10, logbin=False):
    '''Rebins the SEDs along the spectral axis into user-defined binsizes, then combines the fluxes in each bin together according to one of four statistics: average, weighted average, addition, or a user-defined function. Returns a Sed object with an extra attribute 'count' containing the number of flux counts per bin.

    Args:
        aggrseds (iterable): an iterable of Seds, Spectra or AggregateSeds to stack. You cannot mix-and-match Seds and Spectra together in this iterable [unless they are contained in an AggregateSed]
        binsize (tuple): a tuple of the numerical value to bin the spectral axis by and the binsize units, written like (binsize, 'units'). The fluxes within each bin are combined according to the statistic argument. binsize is also the resolution of the stacked SED. By default, 'units' is assumed to be 'Angstroms.'

    Kwargs:
        statistic (str, func): the statistic to use for combining the fluxes in each bin. The possible statistics are:
            'avg' (default) - averages the fluxes within each bin using numpy.average
            'wavg' - computes the weighted average for the fluxes within each bin using the corresponding fluxerrors and numpy.average
            'sum' - sums the fluxes within each bin.
            func - a user-defined function for combining the fluxes in each bin.

        fill (str): switch that decides what to do with bins that have no flux counts in them. There are two options:
            'nan' - the Y-values of unpopulated X-values are assigned numpy.nan
            'remove' - (Default) removes the unpopulated X-Y pair from the stacked SED

        smooth (bool): specifies whether the stacked SED should be smoothed using a boxcar method (True) or not (False)

        smooth_binsize (int): the size of the boxcar. Default is 10.

        log_bin (bool: specifies how to bin the Stack
            False - (default) linear binning
            True - logarithmic binning
            If the Stack's spectral axis ranges over 3 or 4 decades, it is advised to set log_bin = True

    Returns:
        Sed object, with attribute 'count'. Attributes z, xunit and yunit are taken from the first Segment in the list.
            counts - number of flux values combined per binsize. plotting 'counts' against 'x' gives a histogram of the flux counts per spectral bin.

    '''


    if type(binsize) not in (types.FloatType, types.IntType, numpy.float_, numpy.int_):
        raise ValueError('binsize[0] must be of numeric type int or float.')

    if type(smooth) not in (types.BooleanType, numpy.bool_):
        raise ValueError('keyword argument smooth must be \'True\' or \'False\'.')

# Haven't implemented units yet, so we only care about the first element in binsize.
#    if len(binsize) != 2:
#        raise Exception('Argument binsize must be a tuple:'+
#                        '(binsize, \'units\')')
#    if not isinstance(binsize[1], str):
#        raise Exception('units must be a string recognized by the AstroPy.units package')


    # ADD: parameter and algorthim to deal with logarithmic binning

    # making "giant"/global arrays
    giant_spec = numpy.array([])
    giant_flux = numpy.array([])
    giant_fluxerr = numpy.array([]) 
    for xarrs in aggrseds.x:
        giant_spec = numpy.append(giant_spec, xarrs)
    for yarrs in aggrseds.y:
        giant_flux = numpy.append(giant_flux, yarrs)
    for yerrs in aggrseds.yerr:
        giant_fluxerr = numpy.append(giant_fluxerr, yerrs)

    xarr = calc.big_spec(giant_spec, binsize, logbin)

    yarr, xarr, yerrarr, counts = calc.binup(giant_flux, giant_spec, xarr,
                                             statistic, binsize, fill,
                                             giant_fluxerr, logbin=logbin)

    # take first entry of xunit, yunit and z from the aggregate SED
    # for the same attributes in the stacked SED
    xunit = [aggrseds.xunit[0]]
    yunit = [aggrseds.yunit[0]]
    z = aggrseds.z[0]

    # for smoothing.
    # what to do with y-errors?
    if smooth:
        yarr = calc.smooth(yarr, smooth_binsize)

    stacked_sed = Sed(x=xarr, y=yarr, yerr=yerrarr,
                      xunit=xunit, yunit=yunit, z=z)

    setattr(stacked_sed, 'counts', counts)

    return stacked_sed


def create_from_points(points, z=None):
    '''Creates and returns a Sed object from an iterable of PhotometricPoints.
    Args:
        points (iterable): A collection (list, set, tuple) of PhotometricPoints
    Kwargs:
        z (float): The redshift of the Sed. Default is None.'''
    
    sed = Sed()
    for point in points:
        sed.append(point)
    sed.z = z
    return sed


def shift(spec, flux, z, z0):
    '''
    Redshifts the spectrum/SED.

    Args:
        spec (array): Spectral coordinates at observed redshift, z
        flux (array): Flux values at observed redshift, z
        z (float, int): Observed redshift of the SED/spectrum
        z0 (float, int): Target redshift to shift the SED/spectrum to

    Returns:
        spec_z0 (array): Shifted spectral coordinates
        flux_z0 (array): Flux corrected for the intrinsic dimming/brightening due to shifting the SED
    '''

    if z is None:
        raise NoRedshiftError
    if type(z0) not in (types.FloatType, numpy.float_, types.IntType, numpy.int_):
        raise InvalidRedshiftError(0)
    if type(z) not in (types.FloatType, numpy.float_, types.IntType, numpy.int_):
        raise InvalidRedshiftError(0)
    if z0 < 0:
        raise InvalidRedshiftError(1)

    spec = numpy.ma.masked_invalid(spec)
    flux = numpy.ma.masked_invalid(flux)

    spec_z0 = (1 + z0) * spec / (1+z)
    
    z0_total_flux = numpy.trapz(spec_z0, flux)
    z_total_flux = numpy.trapz(spec, flux)
    
    flux_z0 = flux*z_total_flux/z0_total_flux
    
    return numpy.array(spec_z0), numpy.array(flux_z0)


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


def _get_setattr(new_object, old_object):
    for key in old_object.__dict__:
        if key not in new_object.__dict__:
            setattr(new_object, key, old_object.__dict__[key])
