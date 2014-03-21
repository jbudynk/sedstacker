import unittest
import numpy
from sedstacker.sed import Segment, Sed, Spectrum, correct_flux_, shift, _get_setattr
from astLib import astSED as astsed
from sedstacker.exceptions import NoRedshiftError, InvalidRedshiftError


class TestSegment(unittest.TestCase):

    _x = numpy.array([3823.0, 4470.9, 5657.1, 6356.3, 7000.0])
    _y = numpy.array([1.3e-11, 2.56e-11, 7.89e-11, 6.5e-11, 1.2e-10])
    _yerr = numpy.array([1.0e-13, 1.0e-13, 1.0e-13, 1.0e-13, 1e-12])
    _xunit = ['Angstrom']
    _yunit = ['erg/s/cm**2/Angstrom']
    _z = 1.65


    def test_normalize_at_point_sed0(self):
        
        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                   xunit = self._xunit, yunit = self._yunit,
                   z = self._z)

        norm_sed = sed.normalize_at_point(5000.0, 1e-11, norm_operator=0)

        self.assertAlmostEqual(norm_sed[2].y, 3.08203125e-11)
        self.assertAlmostEqual(norm_sed[1].y, 1.0e-11)

    def test_normalize_at_point_sed1(self):
        
        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                   xunit = self._xunit, yunit = self._yunit,
                   z = self._z)

        norm_sed = sed.normalize_at_point(5000.0, 1e-11, norm_operator=1)

        self.assertAlmostEqual(norm_sed[2].y, 6.33e-11)
        self.assertAlmostEqual(norm_sed[1].y, 1e-11)

    def test_normalize_by_int_spectrum(self):
        
        spectrum = Spectrum(x = numpy.linspace(3000.0, 10000.0, num=10000),
                            y = numpy.linspace(1e-13, 1e-11, num=10000))
        norm_spectrum = spectrum.normalize_by_int()

        astlib_spectrum = astsed.SED(wavelength=numpy.linspace(3000.0, 10000.0, num=10000),
                                     flux = numpy.linspace(1e-13, 1e-11, num=10000))
        astlib_spectrum.normalise()


        self.assertEqual(norm_spectrum.y[200], astlib_spectrum.flux[200])
        self.assertAlmostEqual(norm_spectrum.y[200] / norm_spectrum.norm_constant, spectrum.y[200])

    def test_normalize_by_int_sed(self):
        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                   xunit = self._xunit, yunit = self._yunit,
                   z = self._z)
        norm_sed = sed.normalize_by_int()

        norm_const = 1.0/numpy.trapz(abs(self._y),self._x)
        flux = self._y*norm_const
        fluxerr = self._yerr*norm_const

        self.assertEqual(norm_sed.norm_constant, norm_const)
        self.assertEqual(norm_sed[2].y, flux[2])


    def test_norm_at_point_spectrum1(self):
        
        spectrum = Spectrum(x = numpy.linspace(3000.0, 10000.0, num=10000),
                            y = numpy.linspace(1e-13, 1e-11, num=10000))
        
        norm_spectrum = spectrum.normalize_at_point(5000.0, 1e-13)

        assert abs(norm_spectrum.norm_constant - 0.034145) < 0.00001
        self.assertEqual(len(norm_spectrum.yerr), len(norm_spectrum.x))
        self.assert_(numpy.isnan(norm_spectrum.yerr[1]))


    def test_norm_at_point_spectrum2(self):
        
        spectrum = Spectrum(x = range(0, 101),
                            y = range(0, 101))
        
        norm_spectrum = spectrum.normalize_at_point(20, 50, dx=10)

        control_norm_constant = 50.0/20.0
        control_norm_spectrum_y = spectrum.y*control_norm_constant

        self.assertEqual(norm_spectrum.norm_constant, 2.5)
        numpy.testing.assert_array_almost_equal(norm_spectrum.y, control_norm_spectrum_y)


    def test_norm_at_point_spectrum3(self):
        
        spectrum = Spectrum(x = range(0, 101),
                            y = range(0, 101))
        norm_spectrum = spectrum.normalize_at_point(20, 50, dx=50)
        control_avg_flux = 35.0
        control_norm_constant = 50.0/control_avg_flux
        self.assertEqual(norm_spectrum.norm_constant, control_norm_constant)


    def test_shift_sed(self):
        
        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                   xunit = self._xunit, yunit = self._yunit,
                   z = self._z)

        shifted_sed_cfeqTrue = sed.shift(0.1)
        shifted_sed_cfeqTrue_arr = shifted_sed_cfeqTrue._toarray()

        # what sed.shift() should do
        spec_z0 = self._x*(1+0.1)/(1+sed.z)
        zflux = numpy.trapz(self._y, self._x)
        z0flux = numpy.trapz(self._y, spec_z0)
        const = zflux/z0flux

        self.assertEqual(shifted_sed_cfeqTrue.z, 0.1)
        self.assertEqual(isinstance(shifted_sed_cfeqTrue, Sed), True)
        numpy.testing.assert_array_almost_equal(shifted_sed_cfeqTrue_arr[1], self._y*const)

        # not correcting for dimming: correct_flux = False
        shifted_sed_cfeqFalse = sed.shift(0.1, correct_flux = False)
        shifted_sed_cfeqFalse_arr = shifted_sed_cfeqFalse._toarray()

        numpy.testing.assert_array_equal(shifted_sed_cfeqFalse_arr[1], self._y)
        numpy.testing.assert_array_equal(shifted_sed_cfeqFalse_arr[0],shifted_sed_cfeqTrue_arr[0])
        numpy.testing.assert_array_equal(shifted_sed_cfeqFalse_arr[0],spec_z0)


    def test_shift_spectrum(self):

        spectrum = Spectrum(x = numpy.linspace(3000.0, 10000.0, num=10000),
                            y = numpy.linspace(1e-13, 1e-11, num=10000), 
                            z = 1.65)

        shifted_spectrum_cfeqTrue = spectrum.shift(0.1)
        shifted_spectrum_cfeqFalse = spectrum.shift(0.1, correct_flux = False)

        # spec_z0 = spectrum.x*(1+0.1)/(spectrum.z)
        # zflux = numpy.trapz(spectrum.y, spectrum.x)
        # z0flux = numpy.trapz(spectrum.y, spec_z0)
        # const = zflux/z0flux
        const = 1.4673584905660368e-08/3.5350000000000254e-08

        self.assertEqual(shifted_spectrum_cfeqTrue.z, 0.1)
        self.assertAlmostEqual(shifted_spectrum_cfeqTrue.x[9], 1247.8983747) #, delta=1e-6)
        numpy.testing.assert_array_almost_equal(shifted_spectrum_cfeqTrue.y, spectrum.y*const)
        numpy.testing.assert_array_equal(shifted_spectrum_cfeqFalse.x, shifted_spectrum_cfeqTrue.x)
        numpy.testing.assert_array_equal(shifted_spectrum_cfeqFalse.y, spectrum.y)


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


    def test_correct_flux(self):

        spectrum = Spectrum(x=numpy.linspace(3000.0, 10000.0, num=10000),
                            y=numpy.linspace(3e-13, 1e-11, num=10000),
                            z=0.0)

        z_original = 0.1
        correct_flux = correct_flux_(spectrum.x, spectrum.y, spectrum.z, z_original)

        specz0 = spectrum.x * (1+z_original) / (1+spectrum.z)
        tmp, fluxz = shift(specz0, spectrum.y, z_original, spectrum.z)
        control_correct_flux = fluxz

        numpy.testing.assert_array_almost_equal(tmp, spectrum.x)
        numpy.testing.assert_array_almost_equal(control_correct_flux, correct_flux)
        self.assertNotEqual(spectrum.y[5],correct_flux[5])


    def test_get_setattr(self):

        sed = Sed(x=[1,2,3],y=[1,2,3,])
        sed.id = 2051
        sed.stuff = 'check me out'

        sed2 = Sed(x=[4,5,6],y=[4,5,6])
        _get_setattr(sed2,sed)

        self.assertEqual(sed2.id, sed.id)
        self.assertNotEqual(sed2[0].x,sed[0].x)
        self.assertEqual(sed.stuff, sed2.stuff)

        
if __name__ == '__main__':
    unittest.main()
