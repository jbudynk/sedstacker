import numpy
import unittest
import warnings
import logging
from sedstacker.sed import AggregateSed, Sed, Spectrum, Segment, Stack
from sedstacker.exceptions import *


class TestAggregateSed(unittest.TestCase):

    def test_normalize_at_point_seds(self):

        sed1 = Sed(x=numpy.arange(1000,10001,500),
                   y=numpy.arange(1000,10001,500),
                   yerr=numpy.arange(1000,10001,500)*.01)
        sed2 = Sed(x=numpy.arange(10500,20501,500),
                   y=numpy.arange(10500,20501,500),
                   yerr=numpy.arange(10500,20501,500)*.01)
        sed3 = Sed(x=numpy.arange(30000,40001,500),
                   y=numpy.arange(30000,40001,500),
                   yerr=numpy.arange(30000,40001,500)*.01)
        
        aggsed = AggregateSed([sed1,sed2,sed3])
        
        norm_aggsed = aggsed.normalize_at_point(10350,1)
        
        control_norm_constant = 9.523809523809524e-05 # 1./10500.

        self.assertAlmostEqual(norm_aggsed.norm_constant, control_norm_constant)
        # The AggregateSed should
        # retain all Segments after the normalization, even though
        # none of the Segments do not contain the point (10350).
        self.assertEqual(len(norm_aggsed), len(aggsed))

        norm_aggsed = aggsed.normalize_at_point(10251,1)
        self.assertAlmostEqual(norm_aggsed.norm_constant, 1./10500)


    def test_normalize_at_point_spectra(self):

        sed1 = Spectrum(x=numpy.arange(1000,10001),
                   y=numpy.arange(1000,10001),
                   yerr=numpy.arange(1000,10001)*.01)
        sed2 = Spectrum(x=numpy.arange(10500,20501),
                   y=numpy.arange(10500,20501),
                   yerr=numpy.arange(10500,20501)*.01)
        sed3 = Spectrum(x=numpy.arange(30000,40001),
                   y=numpy.arange(30000,40001),
                   yerr=numpy.arange(30000,40001)*.01)
        
        aggsed = AggregateSed([sed1,sed2,sed3])
        
        norm_aggsed = aggsed.normalize_at_point(20000,1,dx=50)
        
        control_norm_constant = 1./20000.

        self.assertAlmostEqual(norm_aggsed.norm_constant, control_norm_constant)
        # The AggregateSed should
        # retain all Segments after the normalization, even though
        # none of the Segments do not contain the point (10350).
        self.assertEqual(len(norm_aggsed), len(aggsed))


    def test_normalize_at_point_segments(self):

        spec1 = Spectrum(x=numpy.arange(1000,10001),
                   y=numpy.arange(1000,10001),
                   yerr=numpy.arange(1000,10001)*.01)
        spec2 = Spectrum(x=numpy.arange(10500,20501),
                   y=numpy.arange(10500,20501),
                   yerr=numpy.arange(10500,20501)*.01)
        spec3 = Spectrum(x=numpy.arange(30000,40001),
                   y=numpy.arange(30000,40001),
                   yerr=numpy.arange(30000,40001)*.01)
        sed1 = Sed(x=numpy.arange(3000,11001,500),
                   y=numpy.arange(300,1101,50),
                   yerr=numpy.arange(300,1101,50)
                   )
        sed2 = Sed(x=numpy.arange(25000,30001,500),
                   y=numpy.arange(2500,3001,50),
                   yerr=numpy.arange(2500,3001,50)*.01)

        aggsed = AggregateSed([spec1,spec2,spec3,sed1,sed2])
        norm_aggsed = aggsed.normalize_at_point(4505.0, 1000.0, dx=50)

        avg = numpy.average(numpy.append(numpy.arange(4456,4555), [450]))
        control_norm_constant = 1000.0/avg

        self.assertAlmostEqual(norm_aggsed.norm_constant, control_norm_constant)


    def test_normalize_by_int(self):

        sed1 = Sed(x=numpy.arange(1000,10001,500),
                   y=numpy.arange(1000,10001,500),
                   yerr=numpy.arange(1000,10001,500)*.01)
        sed2 = Spectrum(x=numpy.arange(3000,9501,10),
                        y=numpy.arange(3000,9501,10),
                        yerr=numpy.arange(3000,9501,10)*.01)
        sed3 = Sed(x=numpy.arange(30000,40001,500),
                   y=numpy.arange(30000,40001,500),
                   yerr=numpy.arange(30000,40001,500)*.01)

        aggsed = AggregateSed([sed1,sed2,sed3])
        
        norm_aggsed = aggsed.normalize_by_int()

        x = numpy.hstack([sed1.x,sed2.x,sed3.x])
        y = numpy.hstack([sed1.y,sed2.y,sed3.y])
        points = []
        for i, point in enumerate(x):
            points.append((x[i],y[i]))
        points = zip(*sorted(points))
        tot_flux = numpy.trapz(points[1], points[0])
        control_norm_constant = 1.0/tot_flux  #1.0/779625000.0

        self.assertAlmostEqual(norm_aggsed.norm_constant, control_norm_constant)
        # The AggregateSed should retain all Segments after the normalization.
        self.assertEqual(len(norm_aggsed), len(aggsed))

        
    def test_shift(self):
        
        sed1 = Sed(x=numpy.arange(1000,10001,500),
                   y=numpy.arange(1000,10001,500),
                   yerr=numpy.arange(1000,10001,500)*.01,)
        sed2 = Spectrum(x=numpy.arange(3000,9501,10),
                        y=numpy.arange(3000,9501,10),
                        yerr=numpy.arange(3000,9501,10)*.01)
        sed3 = Sed(x=numpy.arange(30000,40001,500),
                   y=numpy.arange(30000,40001,500),
                   yerr=numpy.arange(30000,40001,500)*.01)

        aggsed = AggregateSed([sed1,sed2,sed3], z=1.0)

        shifted_aggsed = aggsed.shift(0.0)

        control_x0 = sed1.x/2.0
        control_x1 = sed2.x/2.0
        
        numpy.testing.assert_array_almost_equal(shifted_aggsed[0].x, control_x0)
        numpy.testing.assert_array_almost_equal(shifted_aggsed[1].x, control_x1)


    def test_sorting(self):

        sed1 = Sed(x=[1,5,10,57,60],
                   y=[1,1,1,1,1],
                   yerr=[.1,.1,.1,.1,.1])
        sed2 = Spectrum(x=[0.5,2,15,50,60],
                        y=[2,2,2,2,2],
                        yerr=[.2,.2,.2,.2,.2])
        sed3 = Sed(x=[3,6,9,20],
                   y=[3,3,3,3],
                   yerr=[.3,.3,.3,.3])

        aggsed = AggregateSed([sed1,sed2,sed3])
        control_x = numpy.array([0.5,1,2,3,5,6,9,10,15,20,50,57,60,60])
        control_y = numpy.array([2,1,2,3,1,3,3,1,2,3,2,1,1,2])

        x, y, yerr = aggsed._sorts()

        numpy.testing.assert_array_equal(x, control_x)
        numpy.testing.assert_array_equal(y, control_y)

    
    def test_add_segment(self):
        
        sed1 = Sed(x=[1,5,10,57,60],
                   y=[1,1,1,1,1],
                   yerr=[.1,.1,.1,.1,.1])
        sed2 = Spectrum(x=[0.5,2,15,50,60],
                        y=[2,2,2,2,2],
                        yerr=[.2,.2,.2,.2,.2])
        sed3 = Sed(x=[3,6,9,20],
                   y=[3,3,3,3],
                   yerr=[.3,.3,.3,.3])

        aggsed = AggregateSed([sed1,sed2])
        aggsed.add_segment(sed3)

        self.assertEqual(len(aggsed), 3)
        numpy.testing.assert_array_equal(aggsed[2].x, [3,6,9,20])

        # to see if the class methods are working properly
        # with the added segment
        norm_aggsed = aggsed.normalize_by_int()
        integral = numpy.trapz(numpy.array([2,1,2,3,1,3,3,1,2,3,2,1,1,2]), numpy.array([0.5,1,2,3,5,6,9,10,15,20,50,57,60,60]))
        control_norm_constant = 1.0/integral
        self.assertAlmostEqual(norm_aggsed.norm_constant, control_norm_constant)


if __name__ == '__main__':
    unittest.main()
