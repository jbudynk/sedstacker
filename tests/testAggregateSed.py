import numpy
import unittest
from sedstacker.sed import AggregateSed, Sed, Spectrum, Segment
from sedstacker.exceptions import *


class TestAggregateSed(unittest.TestCase):

    segment1 = Sed()
    segment2 = Spectrum()
    segment3 = Sed()

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


    def test_norm_at_point(self):
        self.assert_(True, "Not implemented yet")


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
        


if __name__ == '__main__':
    unittest.main()
