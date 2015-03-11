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

import logging
import warnings
import numpy
import types

from sedstacker.sed import *

logger=logging.getLogger(__name__)
formatter=logging.Formatter('%(levelname)s:%(message)s')
hndlr=logging.StreamHandler()
hndlr.setFormatter(formatter)
logger.addHandler(hndlr)

class IrisStack(Stack):

    def __init__(self, seds):
        # for i in range(len(seds)):
        #     seds[i] = IrisSed(x=seds[i].x, y=seds[i].y, yerr=seds[i].yerr, z=seds[i].z)
        super(IrisStack, self).__init__(seds)

    def __mul__(self, other):
        for sed in self:
            sed * other
        return self

    def __div__(self, other):
        for sed in self:
            sed / other
        return self

    def __add__(self, other):
        for sed in self:
            sed + other
        return self

    def __sub__(self, other):
        for sed in self:
            sed - other
        return self
        

    def normalize_by_int(self, minWavelength='min', maxWavelength='max', 
                         stats='value', y0=1.0, norm_operator=0, 
                         correct_flux=False, z0=None):

        if isinstance(z0, NUMERIC_TYPES+(types.NoneType,)):
            z0 = [z0]*len(self.segments)
        elif len(z0) != len(self.segments):
            raise ValueError('Length of z0 does not match length of Stack.')

        built_in_stats = {'value':value_, 'avg':avg_, 'median':median_}
        if isinstance(stats, types.StringType):
            if stats not in built_in_stats.keys():
                raise ValueError("Unknown statistic used for normalizing stacks.")
            else:
                stats = built_in_stats[stats]
        else:
            raise ValueError("Statistic must be string")

        excluded = []
        norm_segments = []
        for i, segment in enumerate(self.segments):
            try:
                norm_seg = segment.normalize_by_int(minWavelength=minWavelength,
                                                    maxWavelength=maxWavelength,
                                                    y0=y0, 
                                                    norm_operator=norm_operator,
                                                    correct_flux=correct_flux,
                                                    z0 = z0[i])                
                norm_segments.append(norm_seg)
            except OutsideRangeError:
                # logger.warning(' Excluding Stack[%d] from the normalized Stack.' % self.index(segment))
                logger.warning(' Stack[%(index)d] does not fall within normalization range [%(min)d, %(max)d].'
                               ' Stack[%(index)d] was not normalized.' %
                               {"index": self.index(segment), "min": minWavelength, "max": maxWavelength})
                norm_segments.append(segment)
                excluded.append(segment.id)
            except SegmentError, e:
                # logger.warning(' Excluding Stack[%d] from the normalized Stack' % self.index(segment))
                logger.warning(' Stack[%(index)d] has less than 2 points within the normalization range.'
                               ' Stack[%(index)d] was not normalized.' % {"index": self.index(segment)})
                norm_segments.append(segment)
                excluded.append(segment.id)

        norm_stack = Stack(norm_segments)
        norm_constant = stats(norm_stack)
        
        if not isinstance(norm_constant, types.NoneType):
            norm_segments = []
            for i, segment in enumerate(self.segments):
                try:
                    norm_seg = segment.normalize_by_int(minWavelength=minWavelength,
                                                        maxWavelength=maxWavelength,
                                                        y0=norm_constant, 
                                                        norm_operator=norm_operator,
                                                        correct_flux=correct_flux,
                                                        z0 = z0[i])                
                    norm_segments.append(norm_seg)
                except SegmentError:
                    norm_segments.append(segment)
                except OutsideRangeError:
                    norm_segments.append(segment)
            norm_stack = Stack(norm_segments)
        else:
            pass

        setattr(norm_stack, 'excluded', excluded)
        return norm_stack


    def normalize_at_point(self, x0, y0, stats='value', norm_operator=0, 
                           correct_flux=False, z0=None):

        if isinstance(z0, NUMERIC_TYPES+(types.NoneType,)):
            z0 = [z0]*len(self.segments)
        elif len(z0) != len(self.segments):
            raise ValueError('Length of z0 does not match length of Stack.')

        built_in_stats = {'value':value_, 'avg':avg_, 'median':median_}
        if isinstance(stats, types.StringType):
            if stats not in built_in_stats.keys():
                raise ValueError("Unknown statistic used for normalizing stacks.")
            else:
                stats = built_in_stats[stats]
        else:
            raise ValueError("Statistic must be string")

        excluded = []
        norm_segments = []
        for i, segment in enumerate(self.segments):
            try:
                norm_seg = segment.normalize_at_point(x0, y0, 
                                                      norm_operator=norm_operator,
                                                      correct_flux=correct_flux,
                                                      z0 = z0[i])                
                norm_segments.append(norm_seg)
            except OutsideRangeError:
                logger.warning(' Point (%(x)d, %(y)d) does not fall within spectral range of Stack[%(index)d].'
                               ' Stack[%(index)d] was not normalized.' %
                               {"x": x0, "y": y0, "index": self.index(segment)})
                norm_segments.append(segment)
                excluded.append(segment.id)

        norm_stack = Stack(norm_segments)
        norm_constant = stats(norm_stack)
        
        if not isinstance(norm_constant, types.NoneType):
            norm_segments = []
            for i, segment in enumerate(self.segments):
                try:
                    norm_seg = segment.normalize_at_point(x0, norm_constant,
                                                          norm_operator=norm_operator,
                                                          correct_flux=correct_flux,
                                                          z0 = z0[i])                
                    norm_segments.append(norm_seg)
                except OutsideRangeError:
                    norm_segments.append(segment)
            norm_stack = Stack(norm_segments)
        else:
            pass

        setattr(norm_stack, 'excluded', excluded)
        return norm_stack

    def shift(self, z0, correct_flux=True):

        shifted_segments = []
        excluded = []
        for i, segment in enumerate(self.segments):
            try:
                shifted_segment = segment.shift(z0, correct_flux=correct_flux)
            except NoRedshiftError:
                logger.warning(' Stack[%(index)d] does not have an assigned redshift. This SED has not been shifted.' %
                               {"index": self.index(segment)})
                excluded.append(segment.id)
                shifted_segment = segment
            shifted_segments.append(shifted_segment)

        shifted_stack = Stack(shifted_segments)

        setattr(shifted_stack, 'excluded', excluded)
        return shifted_stack


class IrisSed(Sed):

    def __init__(self, x=[], y=[], yerr=None, z=None, xunit=['Angstrom'], yunit=['erg/s/cm2'], id=None):
        super(IrisSed, self).__init__(x=x, y=y, yerr=yerr, z=z, xunit=xunit, yunit=yunit)
        self.id = id
        self._sort()

    # def __init__(self, sed):
    #     super(IrisSed, self).__init__(x=sed.x, y=sed.y, yerr=sed.yerr, z=sed.z, xunit=sed.xunit, yunit=sed.yunit)
    #     self.id = id
    #     self._sort()

    def __mul__(self, other):
        for point in self:
            point.y = point.y * other
            point.yerr = point.yerr * other
        return self

    def __div__(self, other):
        for point in self:
            point.y = point.y / other
            point.yerr = point.yerr * other
        return self

    def __add__(self, other):
        for point in self:
            point.y = point.y + other
        return self

    def __sub__(self, other):
        for point in self:
            point.y = point.y - other
        return self

    def _sort(self):
        indices = self.x.argsort()
        self.x = self.x[indices]
        self.y = self.y[indices]
        self.yerr = self.yerr[indices]

        # points = []
        # for i, point in enumerate(self.x):
        #     points.append([self.x[i], self.y[i], self.yerr[i]])
        # points = zip(*sorted(points))
        # x = numpy.array(points[0])
        # y = numpy.array(points[1])
        # yerr = numpy.array(points[2])

        # return x, y, yerr

    def normalize_by_int(self, minWavelength='min', maxWavelength='max', 
                         y0=1.0, norm_operator=0, correct_flux=False, z0=None):

        warnings.simplefilter("ignore", UserWarning)

        self._sort()
        spec, flux, fluxerr = self.x, self.y, self.yerr
        flux = numpy.ma.masked_invalid(flux)
        fluxerr = numpy.ma.masked_invalid(fluxerr)
        xunit = self.xunit
        yunit = self.yunit

        if minWavelength == 'min':
            minWavelength=spec.min()
        if maxWavelength == 'max':
            maxWavelength=spec.max()

        lowCut = numpy.greater_equal(spec, minWavelength)
        highCut = numpy.less_equal(spec, maxWavelength)
        totalCut = numpy.logical_and(lowCut, highCut)
        sedWavelengthSlice = spec[totalCut]

        if len(sedWavelengthSlice) == 0:
            raise OutsideRangeError('The normalization range falls outside the spectral range of the Sed.')
        if len(sedWavelengthSlice) < 2:
            raise SegmentError('Sed object must have at least 2 points between minWavelength and maxWavelength.')

        if correct_flux:
            fluxz = correct_flux_(spec, flux, self.z, z0)
            flux = fluxz

        sedFluxSlice = flux[totalCut]
        integral = numpy.trapz(abs(sedFluxSlice), sedWavelengthSlice)
        if norm_operator == 0:
            norm_constant = y0 / integral
            norm_flux = flux * norm_constant
            norm_fluxerr = fluxerr * norm_constant if fluxerr is not None else None
        elif norm_operator == 1:
            norm_constant = y0 - integral
            norm_flux = flux + norm_constant
            norm_fluxerr = fluxerr
        else:
            raise ValueError('Unrecognized norm_operator. keyword \'norm_operator\' must be either 0 (for multiply) or 1 (for addition).')

        norm_segment = IrisSed(x=spec, y=norm_flux, yerr=norm_fluxerr,
                           xunit=xunit, yunit=yunit, z=self.z)
        norm_segment.norm_constant = norm_constant
        norm_segment.norm2flux = integral
        # keep attributes of old sed
        _get_setattr(norm_segment,self)

        return norm_segment

    
    def normalize_at_point(self, x0, y0, norm_operator=0, correct_flux=False, z0=None):

        self._sort()
        spec, flux, fluxerr = self.x, self.y, self.yerr
        xunit = self.xunit
        yunit = self.yunit

        if x0 < self.x.min() or x0 > self.x.max:
            raise OutsideRangeError('The point (%(x)d, %(y)d) falls outside the spectral range of the SED.'
                                    % {"x": x0, "y": y0})

        if correct_flux:
            fluxz = correct_flux_(spec, flux, self.z, z0)
            flux = fluxz  

        interp_self = calc.fast_nearest_interp([x0], spec, flux)
        flux_at_x0 = numpy.float_(interp_self)

        if norm_operator == 0:
            norm_constant = y0 / flux_at_x0
            norm_flux = flux * norm_constant
            norm_fluxerr = fluxerr * norm_constant if fluxerr is not None else None
        elif norm_operator == 1:
            norm_constant = y0 - flux_at_x0
            norm_flux = flux + norm_constant
            norm_fluxerr = fluxerr
        else:
            raise ValueError('Unrecognized norm_operator. keyword \'norm_operator\' must be either 0 (for multiply) or 1 (for addition).')

        norm_sed = Sed(x=spec, y=norm_flux, yerr=norm_fluxerr, xunit=xunit, yunit=yunit, z=self.z)
        norm_sed.norm_constant = norm_constant
        norm_sed.norm2flux = flux_at_x0
        # keep attributes of old sed
        _get_setattr(norm_sed,self)

        return norm_sed

    def shift(self, z0, correct_flux=True):

        self._sort()
        spec, flux, fluxerr = self.x, self.y, self.yerr

        if isinstance(self.z, types.NoneType):
            raise NoRedshiftError
        if (not type(z0) in NUMERIC_TYPES) or (not type(self.z) in NUMERIC_TYPES):
            raise InvalidRedshiftError(0)
        if z0 < 0.0 or self.z < 0.0:
            raise InvalidRedshiftError(1)

        if correct_flux:
            spec_z0, flux_z0 = shift(spec, flux, self.z, z0)
        else:
            spec_z0 = (1 + z0) * self.x / (1+self.z)
            flux_z0 = self.y

        spec = IrisSed(x=spec_z0, y=flux_z0, yerr=fluxerr,
                        xunit=[self.xunit], yunit=[self.yunit], z=z0)
        # keep attributes of old sed
        _get_setattr(spec,self)

        return spec


def avg_(stack):
    norm2fluxs = []
    for sed in stack:
        try:
            norm2fluxs.append(sed.norm2flux)
        except AttributeError:
            pass

    norm_constant = numpy.average(norm2fluxs)
    return norm_constant


def median_(stack):
    norm2fluxs = []
    for sed in stack:
        try:
            norm2fluxs.append(sed.norm2flux)
        except AttributeError:
            pass

    norm_constant = numpy.median(norm2fluxs)
    return norm_constant


def value_(stack):
    return None


def _get_setattr(new_object, old_object):
    for key in old_object.__dict__:
        if key not in new_object.__dict__:
            setattr(new_object, key, old_object.__dict__[key])
