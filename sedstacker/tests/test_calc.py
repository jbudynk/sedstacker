#
import unittest
from sedstacker.exceptions import *
from sedstacker.sed import calc
import numpy


class TestCalc(unittest.TestCase):

    def test_bigspec(self):

        array1 = numpy.linspace(1000,10000, num=100)
        array2 = numpy.arange(100,9900.1, 50)

        # check linear binning
        result = calc.big_spec(numpy.append(array1,array2), 50, False)

        numpy.testing.assert_array_equal(numpy.arange(100,10000+50, 50), result)
        self.assertEqual(result[len(result)-1], 10000)

        # check log binning
        test = calc.big_spec(numpy.append(array1,array2), 0.1, True)
        result = 10**numpy.array([2.,2.1,2.2,2.3,2.4,
                                  2.5,2.6,2.7,2.8,2.9,
                                  3.,3.1,3.2,3.3,3.4,
                                  3.5,3.6,3.7,3.8,3.9,
                                  4.])
        numpy.testing.assert_array_almost_equal(test, result)


    def test_fill_fill(self):
        x = numpy.array([1,5,10,15,50,100])
        mask = [True, True, False, True, False, True]
        new_x = calc.fill_fill(mask, x)

        self.assertEqual(new_x[0], 1)
        self.assert_(numpy.isnan(new_x[2]))


    def test_fill_remove(self):
        x = numpy.array([1,5,10,15,50,100])
        skipit = [1,1,0,1,0,1]
        mask = numpy.ma.make_mask(skipit)
        new_x = calc.fill_remove(mask, x)

        self.assertEqual(new_x[0], 1)
        self.assertEqual(len(new_x), 4)
        self.assertEqual(new_x[2], 15)


    def test_setup_binup_logbinTrue(self):

        x = [1234.,3365.,4407.,5657.,4479.0,6535.]
        y = [1]*len(x)
        yerr = y
        logbin = True
        binsize = 0.1
        xarr = calc.big_spec(x,binsize,logbin)

        x_control = numpy.log10(x)

        x, xarr, y, m_yerr, nx, xbin, yarr, outerr, count, skipit = calc.setup_binup_arrays(y, x, xarr, binsize, yerr, logbin=logbin)

        numpy.testing.assert_array_almost_equal(x, x_control)
        self.assert_(isinstance(xarr, numpy.ndarray))
        self.assertEqual(xbin, 0.05)
        self.assertEqual(len(yarr), nx)
        

    def test_setup_binup_logbinFalse(self):

        x = [1234.,3365.,4407.,5657.,4479.0,6535.]
        y = [1]*len(x)
        yerr = y
        logbin = False
        binsize = 100
        xarr = calc.big_spec(x,binsize,logbin)

        x_control = numpy.array(x)

        x, xarr, y, m_yerr, nx, xbin, yarr, outerr, count, skipit = calc.setup_binup_arrays(y, x, xarr, binsize, yerr, logbin=logbin)

        numpy.testing.assert_array_almost_equal(x, x_control)
        self.assertEqual(xbin, 50.0)
        self.assertEqual(len(m_yerr), len(x))


if __name__ == '__main__':
    unittest.main()
