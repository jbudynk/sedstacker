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
        result = calc.big_spec(numpy.append(array1,array2), 50, True)
        numpy.testing.assert_array_equal(numpy.logspace(numpy.log10(100), numpy.log10(10000), num=(10000-100)/50.), result)


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


if __name__ == '__main__':
    unittest.main()
