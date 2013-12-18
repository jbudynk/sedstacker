import unittest
from sedstacker import sed
from sedstacker.exceptions import *
from sedstacker import calc
import numpy
from math import sqrt


class TestStack(unittest.TestCase):

    x = numpy.linspace(1000,10000, num=10000)
    y = numpy.linspace(1000,10000, num=10000)*1e-10
    yerr = 0.1*y
    z = 0.5

    def test_sanity_check_stack(self):

        # the average-stacked SED of 6 identical SEDs should just be
        # one of the SEDs used in the stack.

        seg1 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg2 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg3 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg4 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg5 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg6 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        
        segments = [seg1,seg2,seg3,seg4,seg5,seg6]
        aggsed = sed.AggregateSed(segments)

        bin = seg1.x[1] - seg1.x[0]

        stacksed = sed.stack(aggsed, bin, 'avg')
        stack_sedarray = stacksed.toarray()

        numpy.testing.assert_array_almost_equal(stack_sedarray[1],aggsed.y[0],decimal=6)
        self.assertEqual(stacksed[3].y,aggsed.y[0][3])
        self.assertEqual(len(stack_sedarray[1]), len(aggsed.y[0]))
        self.assertEqual(stacksed.counts[0], 6)

        stacksed = sed.stack(aggsed, bin, 'wavg')
        stack_sedarray = stacksed.toarray()

        numpy.testing.assert_array_almost_equal(stack_sedarray[1],aggsed.y[0],decimal=6)
        self.assertEqual(stacksed[3].y,aggsed.y[0][3])

        stacksed = sed.stack(aggsed, bin, 'sum')
        stack_sedarray = stacksed.toarray()

        numpy.testing.assert_array_almost_equal(stack_sedarray[1],aggsed.y[0]*6.,decimal=6)
        self.assertEqual(stacksed[3].y,aggsed.y[0][3]*6.)


    def test_no_y_errors_wavg(self):

        seg1 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg2 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg3 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg4 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg5 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg6 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        
        segments = [seg1,seg2,seg3,seg4,seg5,seg6]
        aggsed = sed.AggregateSed(segments)

        bin = seg1.x[1] - seg1.x[0]

        stacksed = sed.stack(aggsed, bin, 'wavg')

        stack_sedarray = stacksed.toarray()

        self.assertEqual(stacksed[3].y,aggsed.y[0][3])
        self.assertEqual(stacksed[3].yerr,sqrt((aggsed.yerr[3][3]**2)*3))
        numpy.testing.assert_array_almost_equal(stack_sedarray[1],aggsed.y[0],decimal=6)
        self.assertEqual(stacksed.counts[0], 3)

    def test_no_y_errors_avg(self):

        seg1 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg2 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg3 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg4 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg5 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg6 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        
        segments = [seg1,seg2,seg3,seg4,seg5,seg6]
        aggsed = sed.AggregateSed(segments)

        bin = seg1.x[1] - seg1.x[0]

        stacksed = sed.stack(aggsed, bin, 'avg')
        stack_sedarray = stacksed.toarray()

        self.assertEqual(stacksed[3].y,aggsed.y[0][3])
        numpy.testing.assert_array_almost_equal(stack_sedarray[1],aggsed.y[0],decimal=6)

    def test_no_y_errors_sum(self):

        seg1 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg2 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg3 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg4 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg5 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg6 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        
        segments = [seg1,seg2,seg3,seg4,seg5,seg6]
        aggsed = sed.AggregateSed(segments)

        bin = seg1.x[1] - seg1.x[0]
        stacksed = sed.stack(aggsed, bin, 'sum')
        stack_sedarray = stacksed.toarray()

        numpy.testing.assert_array_almost_equal(stack_sedarray[1],aggsed.y[0]*6.,decimal=6)
        self.assertEqual(stacksed[3].y,aggsed.y[0][3]*6.)

    def test_stack_spectra(self):

        self.assert_(True, 'Test not implemented yet')


    def test_stack_sed(self):

        self.assert_(True, 'Test not implemented yet')


    def test_stack_AggSed(self):

        self.assert_(True, 'Test not implemented yet')


    def test_wavg(self):

        self.assert_(True, 'Test not implemented yet')


    def test_add(self):

        self.assert_(True, 'Test not implemented yet')


    def test_log_binning(self):

        self.assert_(True, 'Test not implemented yet')


    def test_user_defined_function(self):

        seg1 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg2 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg3 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        segments = [seg1,seg2,seg3]
        aggsed = sed.AggregateSed(segments)

        bin = seg1.x[1] - seg1.x[0]

        def my_average(yarr, yerrarr, counts):
            yout = numpy.average(yarr)
            yerrarr = numpy.average(yerrarr)
            return yout, yerrarr, counts

        stacksed = sed.stack(aggsed, bin, my_average)
        stack_sedarray = stacksed.toarray()

        self.assertEqual(stacksed.counts[0], 3)
        self.assertEqual(stacksed[3].y,aggsed.y[0][3])


if __name__ == '__main__':
    unittest.main()
