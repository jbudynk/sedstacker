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
import os
import os.path
import sedstacker
from sedstacker.exceptions import InvalidRedshiftError, NoRedshiftError
from sedstacker.iris.sed import IrisStack, IrisSed
import numpy
from sedstacker.sed import Stack


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

    def test_redshift_no_z(self):

        sed1 = IrisSed(x=self.x,y=self.y,yerr=self.yerr, id='sed1')
        sed2 = IrisSed(x=numpy.array([2,4,5,8,10]), y=numpy.arange(5)+1.0, yerr=numpy.arange(5)+1.0*0.1, id='sed2')
        y = numpy.array([5.0, 15.0, 7.0, 4.5, 13.5, 10.5])
        x = numpy.array([0.5, 1.5, 3.0, 5.0, 10.5, 21.0])
        sed3 = IrisSed(x=x, y=y, yerr=y*0.1, id='sed3')

        stack = IrisStack([sed1, sed2, sed3])

        shifted_stack = stack.shift(0.0, correct_flux=False)

        self.assertEqual(len(shifted_stack.segments), 3)
        self.assertRaises((InvalidRedshiftError, NoRedshiftError), sed1.shift, -5.0)

        self.assertEqual(shifted_stack.excluded, ['sed1', 'sed2', 'sed3'])

    def test_redshift_correct_flux(self):

        sed1 = IrisSed(x=self.x,y=self.y,yerr=self.yerr, id='sed1', z=0.1)
        sed2 = IrisSed(x=numpy.array([2,4,5,8,10]), y=numpy.arange(5)+1.0, yerr=numpy.arange(5)+1.0*0.1, id='sed2', z=0.1)
        y = numpy.array([5.0, 15.0, 7.0, 4.5, 13.5, 10.5])
        x = numpy.array([0.5, 1.5, 3.0, 5.0, 10.5, 21.0])
        sed3 = IrisSed(x=x, y=y, yerr=y*0.1, id='sed3', z=0.1)

        iris_stack = IrisStack([sed1, sed2, sed3])
        shifted_iris_stack = iris_stack.shift(0.0, correct_flux=True)

        sed1 = IrisSed(x=self.x,y=self.y,yerr=self.yerr, z=0.1)
        sed2 = IrisSed(x=numpy.array([2,4,5,8,10]), y=numpy.arange(5)+1.0, yerr=numpy.arange(5)+1.0*0.1, z=0.2)
        y = numpy.array([5.0, 15.0, 7.0, 4.5, 13.5, 10.5])
        x = numpy.array([0.5, 1.5, 3.0, 5.0, 10.5, 21.0])
        sed3 = IrisSed(x=x, y=y, yerr=y*0.1, z=0.3)

        stack = Stack([sed1, sed2, sed3])

        shifted_stack = stack.shift(0.0, correct_flux=True)

        self.assertEqual(len(shifted_iris_stack.segments), 3)
        numpy.testing.assert_array_almost_equal(shifted_stack[0].x, shifted_iris_stack[0].x)
        numpy.testing.assert_array_almost_equal(shifted_stack[0].y, shifted_iris_stack[0].y)


    def test_outside_norm_ranges(self):

        sed1 = IrisSed(x=[1,2,3,4,5], y=[1,2,3,4,5], id='sed1')
        sed2 = IrisSed(x=[10,20,30,40], y=[1,2,3,4], id='sed2')

        stack = IrisStack([sed1, sed2])

        norm_stack = stack.normalize_by_int(minWavelength=2, maxWavelength=9)

        self.assertEqual(norm_stack.excluded[0], 'sed2')

        sed3 = IrisSed(x=[20,30,40], y=[1,2,3], id='sed3')

        stack.add_segment(sed3)
        norm_stack = stack.normalize_by_int(minWavelength=2, maxWavelength=9)

        numpy.testing.assert_array_equal(norm_stack.excluded, ['sed2', 'sed3'])
        self.assertEqual(len(norm_stack), 3)
