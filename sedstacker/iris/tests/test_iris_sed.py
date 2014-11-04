import unittest
import os
import os.path
import sedstacker
from sedstacker.iris.sed import IrisStack, IrisSed
import numpy

class TestIrisSedStacker(unittest.TestCase):

    x = numpy.array([1,5,10,15,50,100])
    y = numpy.array([1,5,10,15,50,100]) * 0.1
    yerr = numpy.array([1,5,10,15,50,100]) * 0.01

    def test_multiply(self):

        controly = self.y*2.0

        sed1 = IrisSed(x=self.x,y=self.y,yerr=self.yerr)
        sed2 = IrisSed(x=self.x,y=self.y,yerr=self.yerr)

        stack = IrisStack([sed1,sed2]) * 2

        numpy.testing.assert_array_almost_equal(sed1.y, controly)
        numpy.testing.assert_array_almost_equal(stack[0].y, controly)

    def test_normalize_by_int_avg_mult(self):

        sed1 = IrisSed(x=self.x,y=self.y,yerr=self.yerr)
        sed2 = IrisSed(x=numpy.array([2,4,5,8,10]), y=numpy.arange(5)+1.0, yerr=numpy.arange(5)+1.0*0.1)
        y = numpy.array([5.0, 15.0, 7.0, 4.5, 13.5, 10.5])
        x = numpy.array([0.5, 1.5, 3.0, 5.0, 10.5, 21.0])
        sed3 = IrisSed(x=x, y=y, yerr=y*0.1)

        stack = IrisStack([sed1, sed2, sed3])

        # normalize SEDs with avg statistic
        norm_stack = stack.normalize_by_int(minWavelength='min', maxWavelength='max', 
                         stats='avg', y0=1.0, norm_operator=0, 
                         correct_flux=False, z0=None)

        numpy.testing.assert_array_almost_equal(norm_stack[0].y, 0.49234923*sed1.y)
        numpy.testing.assert_array_almost_equal(norm_stack[1].y, 9.846*sed2.y)
        self.assertAlmostEqual(norm_stack[2].norm_constant, 1.1529274)
        

    def test_normalize_by_int_median_mult(self):

        sed1 = IrisSed(x=self.x,y=self.y,yerr=self.yerr)
        sed2 = IrisSed(x=numpy.array([2,4,5,8,10]), y=numpy.arange(5)+1.0, yerr=numpy.arange(5)+1.0*0.1)
        y = numpy.array([5.0, 15.0, 7.0, 4.5, 13.5, 10.5])
        x = numpy.array([0.5, 1.5, 3.0, 5.0, 10.5, 21.0])
        sed3 = IrisSed(x=x, y=y, yerr=y*0.1)

        stack = IrisStack([sed1, sed2, sed3])

        # normalize SEDs with avg statistic
        norm_stack = stack.normalize_by_int(stats='median')

        numpy.testing.assert_array_almost_equal(norm_stack[0].y, 0.4270427*sed1.y)
        numpy.testing.assert_array_almost_equal(norm_stack[1].y, 8.54*sed2.y)
        self.assertAlmostEqual(norm_stack[2].norm_constant, 1.0)
        
    
    def test_normalize_by_int_avg_add(self):

        sed1 = IrisSed(x=self.x,y=self.y,yerr=self.yerr)
        sed2 = IrisSed(x=numpy.array([2,4,5,8,10]), y=numpy.arange(5)+1.0, yerr=numpy.arange(5)+1.0*0.1)
        y = numpy.array([5.0, 15.0, 7.0, 4.5, 13.5, 10.5])
        x = numpy.array([0.5, 1.5, 3.0, 5.0, 10.5, 21.0])
        sed3 = IrisSed(x=x, y=y, yerr=y*0.1)

        stack = IrisStack([sed1, sed2, sed3])

        # normalize SEDs with avg statistic
        norm_stack = stack.normalize_by_int(stats='avg', norm_operator=1)

        numpy.testing.assert_array_almost_equal(norm_stack[0].y, 0 - 253.8 + sed1.y)
        numpy.testing.assert_array_almost_equal(norm_stack[1].y, 221.15 + sed2.y)
        self.assertAlmostEqual(norm_stack[2].norm_constant, 32.65)


    def test_normalize_at_point_avg_mult(self):

        sed1 = IrisSed(x=self.x,y=self.y,yerr=self.yerr)
        sed2 = IrisSed(x=numpy.array([2,4,5,8,10]), y=numpy.arange(5)+1.0, yerr=numpy.arange(5)+1.0*0.1)
        y = numpy.array([5.0, 15.0, 7.0, 4.5, 13.5, 10.5])
        x = numpy.array([0.5, 1.5, 3.0, 5.5, 10.5, 21.0])
        sed3 = IrisSed(x=x, y=y, yerr=y*0.1)

        stack = IrisStack([sed1, sed2, sed3])

        # normalize SEDs with avg statistic
        norm_stack = stack.normalize_at_point(5.0, 1.0, stats='avg', norm_operator=0)

        numpy.testing.assert_array_almost_equal(norm_stack[0].y, (8/3.)/0.5*sed1.y)
        numpy.testing.assert_array_almost_equal(norm_stack[1].y, (8/3.)/3.0*sed2.y)
        self.assertAlmostEqual(norm_stack[2].norm_constant, (8/3.)/4.5)

        
