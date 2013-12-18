#
import unittest
import numpy
from sedstacker.sed import Segment, Sed, Spectrum
from astLib import astSED as astsed
from sedstacker.exceptions import NoRedshiftError, InvalidRedshiftError


class TestSegment(unittest.TestCase):

    _x = numpy.array([3823.0, 4470.9, 5657.1, 6356.3])
    _y = numpy.array([1.3e-11, 2.56e-11, 7.89e-11, 6.5e-11])
    _yerr = numpy.array([1.0e-13, 1.0e-13, 1.0e-13, 1.0e-13])
    _xunit = ['Angstrom']
    _yunit = ['erg/s/cm**2/Angstrom']
    _z = 1.65


#    def test_normalize_at_point(self):
        
#        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
#                   xunit = self._xunit, yunit = self._yunit,
#                   z = self._z)

#        norm_sed = sed.normalize_at_point(5000.0, 1e-11)

#        self.assert_((norm_sed.x == 5000.0), (norm_sed.y == 1.0) == 5000.0, 1e-11)


    def test_normalize_by_int_spectrum(self):
        
        spectrum = Spectrum(x = numpy.linspace(3000.0, 10000.0, num=10000),
                            y = numpy.linspace(1e-13, 1e-11, num=10000))
        norm_spectrum = spectrum.normalize_by_int()

        astlib_spectrum = astsed.SED(wavelength = numpy.linspace(3000.0, 10000.0, num=10000),
                                     flux = numpy.linspace(1e-13, 1e-11, num=10000))
        astlib_spectrum.normalise()


        self.assertEqual(norm_spectrum.y[200], astlib_spectrum.flux[200])
        self.assertAlmostEqual(norm_spectrum.y[200] / norm_spectrum.norm_constant, spectrum.y[200])

    def test_normalize_by_int_sed(self):
        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                   xunit = self._xunit, yunit = self._yunit,
                   z = self._z)
        norm_sed = sed.normalize_by_int()

        astlib_sed = astsed.SED(wavelength = self._x, flux = self._y)
        astlib_sed.normalise()

        norm_sed_cfTrue = sed.normalize_by_int(correct_flux=True, z0 = 0.5)
        self.assertNotEqual(norm_sed_cfTrue.norm_constant, norm_sed.norm_constant)


        self.assertEqual(norm_sed[2].y, astlib_sed.flux[2])
        self.assertAlmostEqual(norm_sed[1].y / norm_sed.norm_constant,  sed[1].y)


    def test_norm_at_point_spectrum(self):
        
        spectrum = Spectrum(x = numpy.linspace(3000.0, 10000.0, num=10000),
                            y = numpy.linspace(1e-13, 1e-11, num=10000))
        
        norm_spectrum = spectrum.normalize_at_point(5000.0, 1e-13)

        self.assertEqual('%.4f' % norm_spectrum.norm_constant, repr(0.0342))
        self.assertEqual(len(norm_spectrum.yerr), len(norm_spectrum.x))
        self.assert_(numpy.isnan(norm_spectrum.yerr[1]))
        

    def test_shift_sed(self):
        
        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                   xunit = self._xunit, yunit = self._yunit,
                   z = self._z)

        shifted_sed_cfeqTrue = sed.shift(0.1)
        #shifted_sed_cfeqFalse = sed.shift(0.1, correct_flux = False)

        self.assertEqual(shifted_sed_cfeqTrue.z, 0.1)
        self.assertEqual(isinstance(shifted_sed_cfeqTrue, Sed), True)


    def test_shift_spectrum(self):

        spectrum = Spectrum(x = numpy.linspace(3000.0, 10000.0, num=10000),
                            y = numpy.linspace(1e-13, 1e-11, num=10000), 
                            z = 1.65)

        shifted_spectrum_cfeqTrue = spectrum.shift(0.1)
        #shifted_spectrum_cfeqFalse = sed.shift(0.1, correct_flux = False)

        self.assertEqual(shifted_spectrum_cfeqTrue.z, 0.1)


    def test_shift_spectrum_raise_InvalidRedshift(self):

        spectrum = Spectrum(x = numpy.linspace(3000.0, 10000.0, num=10000),
                            y = numpy.linspace(1e-13, 1e-11, num=10000), 
                            z = 1.65)

        self.assertRaises(InvalidRedshiftError, spectrum.shift, -1.2)
        self.assertRaises(InvalidRedshiftError, spectrum.shift, 'oh hey')

    def test_shift_raise_NoRedshift(self):

        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                   xunit = self._xunit, yunit = self._yunit,
                   z = None)

        self.assertRaises(NoRedshiftError, sed.shift, 0.9)

        
if __name__ == '__main__':
    unittest.main()
