import numpy
import unittest
import warnings
import logging
import matplotlib.pyplot as plt
from sedstacker.sed import AggregateSed, Sed, Spectrum, Segment
from sedstacker.exceptions import *


class TestAggregateSed(unittest.TestCase):

    def testAggregateSed__init__(self):

        segment1 = Sed()
        segment2 = Spectrum()
        segment3 = Sed()

        aggsed = AggregateSed([segment1, segment2, segment3])
        
        self.assertEqual(len(aggsed), 3)


    def testAggregateSed__init__raiseNotASegment(self):
        
        segment1 = Spectrum()
        segment2 = Sed()
        segment3 = [[1,2,3],[1,2,3]]

        self.assertRaises(NotASegmentError, AggregateSed, [segment1, segment2, segment3])


    def test_arrays1(self):
        
        segment1 = Sed()
        segment2 = Spectrum()
        segment3 = Sed()

        aggsed = AggregateSed([segment1, segment2, segment3])

        numpy.testing.assert_array_equal(aggsed.x[1], numpy.array([]))
        numpy.testing.assert_array_equal(aggsed.x[0], numpy.array([]))
        self.assertEqual(len(aggsed), 3)


    def test_arrays2(self):
        
        segment1 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10))
        segment2 = Sed()
        segment3 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500))

        aggsed = AggregateSed([segment1, segment2, segment3])

        self.assertEqual(len(aggsed), 3)
        self.assertAlmostEqual(aggsed[0].y[10], 1100)


    def test_arrays_for_z(self):

        segment1 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10), z=1.65)
        segment2 = Spectrum(z=1.0)
        segment3 = Spectrum(z=0.1)

        aggsed = AggregateSed([segment1, segment2, segment3])

        self.assertEqual(aggsed.z[1], 1.0)
        self.assertEqual(len(aggsed.z), 3)

    def test_shift(self):

        segment1 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            z = 1.0)
        segment2 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       z = 0.5)
        segment3 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       z = 0.35)

        aggsed = AggregateSed([segment1, segment2, segment3])

        shift_aggsed = aggsed.shift(0.4)

        self.assertEqual(shift_aggsed.z[0], shift_aggsed.z[1])
        self.assertEqual(shift_aggsed.z[0], 0.4)
        self.assertAlmostEqual(shift_aggsed.x[0][10], 770.0)


    def test_shift_NoRedshift(self):

        segment1 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            z=1.0)
        segment2 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500))
        segment3 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       z = 0.35)

        aggsed = AggregateSed([segment1, segment2, segment3])

        shift_aggsed = aggsed.shift(0.5)

        self.assertEqual(len(shift_aggsed), 2)
        self.assertEqual(shift_aggsed.x[0][1], 757.5)
        self.assertEqual('%.2f' % shift_aggsed.x[1][1], repr(1666.67))


    def test_norm_by_int(self):
        
        segment1 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=1.0)
        segment2 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01)
        segment3 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01,
                       z = 0.35)
        segment4 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=1.0)

        aggsed = AggregateSed([segment1, segment2, segment3, segment4])

        norm_aggsed = aggsed.normalize_by_int()

        self.assertAlmostEqual(norm_aggsed.segments[0].norm_constant, 2.0288029e-08)


    def test_norm_by_int_correct_flux(self):
        
        segment1 = Spectrum(x=numpy.arange(1000,10000,10),
                            y=numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=0.0)
        segment2 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01,
                       z=0.0)
        segment3 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01,
                       z = 0)
        segment4 = Spectrum(x=numpy.arange(1000,10000,10),
                            y=numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=0)

        aggsed = AggregateSed([segment1, segment2, segment3, segment4])

        norm_aggsed = aggsed.normalize_by_int(correct_flux=True, z0=[1,.5,.1,1])

        aggsed[0].z=1.0
        aggsed[1].z=0.5
        aggsed[2].z=0.1
        aggsed[3].z=1.0
        rf_aggsed=aggsed.shift(0.0)
        rf_aggsed0=rf_aggsed[0].shift(1.0, correct_flux=False)
        rf_aggsed1=rf_aggsed[1].shift(0.5, correct_flux=False)
        rf_aggsed2=rf_aggsed[2].shift(0.1, correct_flux=False)
        rf_aggsed3=rf_aggsed[3].shift(1.0, correct_flux=False)
        rf_aggsed=AggregateSed([rf_aggsed0,rf_aggsed1,rf_aggsed2,rf_aggsed3])
        control_norm_aggsed = rf_aggsed.normalize_by_int()

        self.assertAlmostEqual(norm_aggsed.segments[0].norm_constant, control_norm_aggsed[0].norm_constant)
        self.assertAlmostEqual(norm_aggsed[1].norm_constant,1.4939309057e-08)
        numpy.testing.assert_array_almost_equal(control_norm_aggsed[0].y,norm_aggsed[0].y)
        sed=norm_aggsed[1].toarray()
        control_sed=control_norm_aggsed[1].toarray()
        numpy.testing.assert_array_almost_equal(control_sed[1],sed[1])
        numpy.testing.assert_array_almost_equal(control_sed[0],sed[0])


    def test_norm_at_point(self):

        segment1 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=1.0)
        segment2 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01)
        segment3 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01,
                       z = 0.35)
        segment4 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=1.0)

        aggsed = AggregateSed([segment1, segment2, segment3, segment4])
        norm_aggsed = aggsed.normalize_at_point(5000,1000)

        sedarray = segment3.toarray()
        control_norm_aggsed_segment3 = sedarray[1]*0.2

        self.assertEqual(norm_aggsed[1][8].y, 1000)
        sedarray = norm_aggsed[2].toarray()
        numpy.testing.assert_array_almost_equal(sedarray[1], control_norm_aggsed_segment3)
        self.assertEqual(norm_aggsed[0].norm_constant, 0.2)


    def test_norm_at_point_correct_flux(self):

        segment1 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=0)
        segment2 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01,
                       z=0.0)
        segment3 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01,
                       z = 0)
        segment4 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=0)

        aggsed = AggregateSed([segment1, segment2, segment3, segment4])
        norm_aggsed = aggsed.normalize_at_point(5000,1000,
                                                correct_flux=True,
                                                z0=[1.0,0.5,0.35,1.0])


        self.assertAlmostEqual(norm_aggsed[2].norm_constant, 0.148148148)
        self.assertAlmostEqual(norm_aggsed[3].norm_constant, 0.1)


    def test_remove_segment(self):
        
        segment1 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=1.0)
        segment2 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01)
        segment3 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01,
                       z = 0.35)
        segment4 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=1.0)

        aggsed = AggregateSed([segment1, segment2, segment3, segment4])

        aggsed.remove_segment(segment1)

        self.assertEqual(len(aggsed), 3)
        self.assertEqual(len(aggsed.x[0]), len(numpy.arange(1000,10000,500)))

    def test_add_segment(self):

        segment1 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=1.0)
        segment2 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01)
        segment3 = Sed(x=numpy.arange(1000,10000,500),
                       y=numpy.arange(1000,10000,500),
                       yerr=numpy.arange(1000,10000,500)*.01,
                       z = 0.35)
        segment4 = Spectrum(x = numpy.arange(1000,10000,10),
                            y = numpy.arange(1000,10000,10),
                            yerr=numpy.arange(1000,10000,10)*.01,
                            z=1.0)

        aggsed = AggregateSed([segment1, segment2, segment3])

        aggsed.add_segment(segment4)

        self.assertEqual(len(aggsed), 4)
        self.assertEqual(len(aggsed.x[3]), len(segment4.x))
        
        aggsed.add_segment(segment3)

        self.assertEqual(len(aggsed), 5)
        self.assertEqual(len(aggsed.x[4]), len(segment3))
        

#        sed2=norm_aggsed[1].toarray()
#        sed3=norm_aggsed[2].toarray()
#        plt.loglog(norm_aggsed[0].x,norm_aggsed[0].x,'r')
#        plt.loglog(sed2[0],sed2[1],'ko')
#        plt.loglog(sed3[0],sed3[1],'go')
#        plt.loglog(norm_aggsed[3].x,norm_aggsed[3].x,'b')
#        plt.show()

if __name__ == '__main__':
    unittest.main()
