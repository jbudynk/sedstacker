import numpy
import unittest
from sedstacker.exceptions import SegmentError
from sedstacker.sed import Sed, PhotometricPoint, find_range
from math import log10


class TestSed(unittest.TestCase):

    _x = numpy.array([3823.0, 4477.9, 5657.1, 6370.0])
    _y = numpy.array([1.3e-11, 2.56e-11, 7.89e-11, 6.5e-11])
    _yerr = numpy.array([1.0e-13, 1.0e-13, 1.0e-13, 1.0e-13])
    _xunit = ['Angstrom']
    _yunit = ['erg/s/cm**2/Angstrom']
    _z = 1.65

    def test__init__(self):

        sed = Sed(x = numpy.array([3823.0, 44770.9, 5657.1]),
                  y = numpy.array([1.3e-11, 2.56e-11, 7.89e-11]),
                  yerr = numpy.array([1.0e-13, 1.0e-13, 1.0e-13]),
                  xunit = ['Angstrom'],
                  yunit = ['erg/s/cm**2/Angstrom'],
                  z = 1.65)

        self.assertAlmostEqual(sed[0].x, 3823.0)
        self.assert_(sed[1].xunit == 'Angstrom')
        self.failUnlessEqual(len(sed), 3)
        self.assertEqual(len(sed.__dict__), 1)
        self.assertEqual(sed.z, 1.65)


    def test_add_point(self):
        
        sed = Sed()
        point = PhotometricPoint(x=1.0, y=1.0)

        sed.add_point(point)

        self.assertEqual(sed[0].x, 1.0)
        self.assertEqual(sed[0].y, 1.0)
        self.assert_(numpy.isnan(sed[0].yerr))
        self.assertEqual(len(sed[0].__dict__), 6, 'length of sed[0]')

        sed.add_point(PhotometricPoint(x=2.0, y=-2.0))

        self.assertEqual(sed[0].x, 1.0)
        self.assertEqual(sed[1].y, -2.0)
        self.assertEqual(len(sed[1].__dict__), 6, 'length of sed[2]')


    def test_remove_point(self):
        
        sed = Sed()
        point1 = PhotometricPoint(x=1.0, y=1.0)
        point2 = PhotometricPoint(x=2.0, y=-2.0)

        sed.add_point(point1)
        sed.add_point(point2)

        self.assert_(len(sed) == 2)

        sed.remove_point(0)

        self.assert_(len(sed) == 1)
        self.assertEqual(sed[0].x, 2.0)


    def test_mask_point(self):
        
        sed = Sed(x=[0.0], y=[0.0])
        point1 = PhotometricPoint(x=1.0, y=1.0)
        point2 = PhotometricPoint(x=2.0, y=-2.0)

        sed.add_point(point1)
        sed.add_point(point2)

        sed.mask_point(1)
        self.assert_(sed[1].mask)
        
        sed.unmask_point(1)
        self.failIf(sed[1].mask)


    def test_add_sed(self):
        
        sed1 = Sed()
        sed2 = Sed(x = self._x, y = self._y, yerr = self._yerr,
                   xunit = self._xunit, yunit = self._yunit,
                   z = self._z)
        
        sed1.add_segment(sed2)

        self.assertEqual(len(sed1), 4)
        self.assertEqual(sed1[0].x, self._x[0])

        sed3 = Sed(x = self._x, y = self._y, yerr = self._yerr,
                   xunit = self._xunit, yunit = self._yunit,
                   z = self._z)
        sed1.add_segment(sed2, sed3)

        self.assertEqual(len(sed1), 12)
        self.assertEqual(sed1[7].y, self._y[3])
        
        total = 0
        for points in sed1:
            total += points.x
        self.assertEqual(total, sum(self._x)*3)


    def test_shift(self):
        
        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                  xunit = self._xunit, yunit = self._yunit,
                  z = self._z)
        
        shifted_sed_cfTrue = sed.shift(0.1)
        
        self.assertEqual(shifted_sed_cfTrue.z, 0.1)
        self.assertAlmostEqual('%.2f' % shifted_sed_cfTrue[1].x, repr(1858.75))


    def test_normalize_by_int(self):

        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                  xunit = self._xunit, yunit = self._yunit,
                  z = self._z)

        norm_sed = sed.normalize_by_int()

        self.assert_(hasattr(norm_sed, 'norm_constant'))


    def test_normalize_by_int_raise_SegmentError(self):

        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                  xunit = self._xunit, yunit = self._yunit,
                  z = self._z)

        self.assertRaises(SegmentError, sed.normalize_by_int, 4000.0, 5000.0)


    def test_toarray(self):

        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                  xunit = self._xunit, yunit = self._yunit,
                  z = self._z)

        sedarray = sed.toarray()
        self.assertEqual(sedarray[1][0], self._y[0])


    def test_xyyerr_properties(self):
        
        sed = Sed(x = self._x, y = self._y, yerr = self._yerr,
                  xunit = self._xunit, yunit = self._yunit,
                  z = self._z)

        numpy.testing.assert_array_equal(sed.x, self._x)
        numpy.testing.assert_array_equal(sed.y, self._y)
        numpy.testing.assert_array_equal(sed.yerr, self._yerr)

        sed.remove_point(0)
        
        numpy.testing.assert_array_equal(sed.x, self._x[1:])
        numpy.testing.assert_array_equal(sed.y, self._y[1:])
        numpy.testing.assert_array_equal(sed.yerr, self._yerr[1:])

        p=PhotometricPoint(240000.,4.5e-9)
        sed.add_point(p)

        numpy.testing.assert_array_equal(sed.x, numpy.array([4477.9, 5657.1, 6370.0, 240000.]))

        sed.yerr = 1e-10
        numpy.testing.assert_array_equal(sed.yerr, numpy.array([1e-10]*len(sed)))
        with self.assertRaises(AssertionError):
            sed.yerr = [1e-13]
        


if __name__ == '__main__':
    unittest.main()
