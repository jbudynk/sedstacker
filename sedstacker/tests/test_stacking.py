#!/usr/bin/env python
#
#  Copyright (C) 2015  Smithsonian Astrophysical Observatory
#
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License along
#  with this program; if not, write to the Free Software Foundation, Inc.,
#  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
#

import unittest
import sedstacker
from sedstacker import sed
from sedstacker.exceptions import *
from sedstacker import calc
from sedstacker.io import load_cat
import numpy
import os
from math import sqrt


test_directory = os.path.dirname(sedstacker.__file__)+"/tests/resources/"

rootdir = os.path.dirname(sedstacker.__file__)
sed_filename = rootdir+"/tests/resources/phot-cat-mags.ascii"

column_map = {"ucfht":(3823.0,"AA","mag","errU"),
              "Bsub":(4459.7,"AA","mag","errB"),
              "Vsub":(5483.8,"AA","mag","errV"),
              "gsub":(4779.6,"AA","mag","errg"),
              "rsub":(6295.1,"AA","mag","errR"),
              "isub":(7640.8,"AA","mag","errI"),
              "zsub":(9036.9,"AA","mag","errz"),
              "j":(12491.0,"AA","mag","errJ"),
              "Ks":(21590.4,"AA","mag","errK"),
              "irac3.6":(36000.0,"AA","mag","err3.6"),
              "irac4.5":(45000.0,"AA","mag","err4.5"),
              "irac5.8":(58000.0,"AA","mag","err5.8"),
              "irac8.0":(80000.0,"AA","mag","err8.0"),
              "mips24.0":(240000.0,"AA","mag","err24")
              }

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
        numpy.testing.assert_array_almost_equal(stacksed.y,aggsed.y[0],decimal=6)
        self.assertEqual(stacksed[3].y,aggsed.y[0][3])
        self.assertEqual(len(stacksed.y), len(aggsed.y[0]))
        self.assertEqual(stacksed.counts[0], 6)

        stacksed = sed.stack(aggsed, bin, 'wavg')
        numpy.testing.assert_array_almost_equal(stacksed.y,aggsed.y[0],decimal=6)
        self.assertEqual(stacksed[3].y,aggsed.y[0][3])

        stacksed = sed.stack(aggsed, bin, 'sum')
        numpy.testing.assert_array_almost_equal(stacksed.y,aggsed.y[0]*6.,decimal=6)
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

        # because there are seds with no y-errors, stack() will
        # use 'avg' instead of 'wavg'.
        self.assertEqual(stacksed[3].y,aggsed.y[0][3])
        self.assertEqual(stacksed[3].yerr,0.0)
        numpy.testing.assert_array_almost_equal(stacksed.y,aggsed.y[0],decimal=6)
        self.assertEqual(stacksed.counts[0], 6)


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

        self.assertEqual(stacksed[3].y,aggsed.y[0][3])
        numpy.testing.assert_array_almost_equal(stacksed.y,aggsed.y[0],decimal=6)


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
        stack_sedarray = stacksed._toarray()

        numpy.testing.assert_array_almost_equal(stacksed.y,aggsed.y[0]*6.,decimal=6)
        self.assertEqual(stacksed[3].y,aggsed.y[0][3]*6.)
        self.assertEqual(len(stacksed.xunit), len(stacksed.x))


    def test_stack_segments(self):

        seg1 = sed.Sed(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg2 = sed.Sed(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg3 = sed.Sed(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg4 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg5 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg6 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)

        bin = seg1.x[1] - seg1.x[0]

        aggsed = [seg1,seg2,seg3,seg4,seg5,seg6]
        stacked_seds = sed.stack(aggsed, bin, 'wavg')

        self.assertEqual(stacked_seds.y[3],aggsed[0].y[3])
        self.assertEqual(stacked_seds.yerr[3], 0.0)    # variance of identical values is 0.0
        numpy.testing.assert_array_almost_equal(stacked_seds.y[0],aggsed[0].y[0],decimal=6)

        self.assertEqual(stacked_seds.counts[0], 6)


    def test_stack_list_of_aggregateseds(self):

        seg1 = sed.Sed(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg2 = sed.Sed(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg3 = sed.Sed(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg4 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg5 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)
        seg6 = sed.Spectrum(x=self.x,y=self.y,yerr=self.yerr,z=self.z)

        bin = seg1.x[1] - seg1.x[0]

        aggsed = sed.AggregateSed([seg1,seg2,seg3,seg4,seg5,seg6])
        stacked_seds = sed.stack([aggsed,aggsed], bin, 'avg')

        self.assertEqual(stacked_seds.y[3],aggsed[0].y[3])
        numpy.testing.assert_array_almost_equal(stacked_seds.y[0],aggsed[0].y[0],decimal=6)
        self.assertEqual(stacked_seds.counts[0], 12)

    def test_stack_seds(self):

        aggsed = load_cat(sed_filename, column_map)
        stacked_seds = sed.stack(aggsed, 50, 'avg')
        control_x = numpy.array([3823., 4473., 4773.,
                                 5473., 6273., 7623.,
                                 9023., 12473., 21573.,
                                 36023., 45023., 58023.,
                                 80023., 240023.])
        control_y = numpy.array([23.9516667, 23.498333333,
                                 23.45166667, 23.071666667,
                                 22.71166667, 22.414,
                                 21.97833333, 21.25,
                                 20.295, 19.61,
                                 19.36, 19.07833333,
                                 18.88, 17.565])

        numpy.testing.assert_allclose(stacked_seds.x, control_x)
        numpy.testing.assert_allclose(stacked_seds.y, control_y)

        stacked_seds = sed.stack(aggsed, 50, 'wavg')
        control_x = numpy.array([3823., 4473., 4773.,
                                 5473., 6273., 7623.,
                                 9023., 12473., 21573.,
                                 36023., 45023., 58023.,
                                 80023., 240023.])
        control_y = numpy.array([23.9516667, 23.498333333,
                                 23.45166667, 23.071666667,
                                 22.71166667, 22.414,
                                 21.97833333, 21.25,
                                 20.295, 19.61,
                                 19.36, 19.07833333,
                                 18.88, 17.565])

        numpy.testing.assert_allclose(stacked_seds.x, control_x)
        numpy.testing.assert_allclose(stacked_seds.y, control_y)


    def test_log_binning(self):

        aggsed = load_cat(sed_filename, column_map)
        stacked_seds = sed.stack(aggsed, 0.2, 'avg', logbin=True)

        control_x = 10**numpy.array([3.6,3.8,4.0,4.4,4.6,4.8,5.0,5.4])
        control_y = numpy.array([23.63388888888888, 22.751176470588234,
                                 21.614166666666, 20.295,
                                 19.485, 19.0783333333,
                                 18.88, 17.565])
        numpy.testing.assert_allclose(stacked_seds.x, control_x)
        numpy.testing.assert_allclose(stacked_seds.y, control_y)


    def test_user_defined_function(self):

        seg1 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg2 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        seg3 = sed.Spectrum(x=self.x,y=self.y,z=self.z)
        segments = [seg1,seg2,seg3]
        aggsed = sed.AggregateSed(segments)

        bin = seg1.x[1] - seg1.x[0]

        def my_average(yarr, yerrarr, nans):
            yout = numpy.average(yarr)
            yerrarr = numpy.average(yerrarr)
            counts = len(yarr)
            return yout, yerrarr, counts

        stacksed = sed.stack(aggsed, bin, my_average)

        self.assertEqual(stacksed.counts[0], 3)
        self.assertEqual(stacksed[3].y,aggsed.y[0][3])


    def test_stack_one_segment(self):
        
        seg1 = sed.Sed(x=numpy.linspace(1000,10000, num=100),
                       y=numpy.linspace(1000,10000, num=100)*0.001)
        seg2 = sed.Sed(x=numpy.linspace(1000,10000, num=100),
                       y=numpy.linspace(1000,10000, num=100)*0.001)
        seg1.add_segment(seg2)
        
        bin = seg1.x[1] - seg1.x[0]

        stackedsed = sed.stack([seg1], bin, 'avg')

        self.assertEqual(stackedsed[3].y,seg1[3].y)
        self.assertEqual(stackedsed.counts[0],2)


if __name__ == '__main__':
    unittest.main()
