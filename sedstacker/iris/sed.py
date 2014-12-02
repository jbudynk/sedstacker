#!/usr/bin/env python

import logging
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
            except SegmentError:
                logger.warning(' Excluding Stack[%d] from the normalized Stack' % self.index(segment))
                pass

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
                    logger.warning(' Excluding Stack[%d] from the normalized Stack' % self.index(segment))
                    pass
            norm_stack = Stack(norm_segments)
        else:
            pass

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

        norm_segments = []
        for i, segment in enumerate(self.segments):
            try:
                norm_seg = segment.normalize_at_point(x0, y0, 
                                                      norm_operator=norm_operator,
                                                      correct_flux=correct_flux,
                                                      z0 = z0[i])                
                norm_segments.append(norm_seg)
            except OutsideRangeError:
                logger.warning(' Excluding Stack[%d] from the normalized Stack.' % self.index(segment))
                pass
            except SegmentError, e:
                logger.warning(' Excluding Stack[%d] from the normalized Stack' % self.index(segment))
                pass

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
                except SegmentError:
                    logger.warning(' Excluding Stack[%d] from the normalized Stack' % self.index(segment))
                    pass
            norm_stack = Stack(norm_segments)
        else:
            pass

        return norm_stack

    def shift(self, z0, correct_flux=True):

        shifted_segments = []

        for i, segment in enumerate(self.segments):
            try:
                shifted_segment = segment.shift(z0, correct_flux=correct_flux)
            except NoRedshiftError:
                logger.warning(' One or more SEDs do not have an assigned redshift. These SEDs have not been shifted.')
                shifted_segment = segment
            shifted_segments.append(shifted_segment)

        shifted_stack = Stack(shifted_segments)

        return shifted_stack


class IrisSed(Sed):

    def __init__(self, x=[], y=[], yerr=None, z=None, xunit=['Angstrom'], yunit=['erg/s/cm2']):
        super(IrisSed, self).__init__(x=x, y=y, yerr=yerr, z=z, xunit=xunit, yunit=yunit)

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

    def normalize_by_int(self, minWavelength='min', maxWavelength='max', 
                         y0=1.0, norm_operator=0, correct_flux=False, z0=None):
        
        spec = self.x
        flux = numpy.ma.masked_invalid(self.y)
        fluxerr = numpy.ma.masked_invalid(self.yerr)
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
        
        spec = self.x
        flux = self.y
        fluxerr = self.yerr
        xunit = self.xunit
        yunit = self.yunit

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


def avg_(stack):
    norm2fluxs = []
    for sed in stack:
        norm2fluxs.append(sed.norm2flux)

    norm_constant = numpy.average(norm2fluxs)
    return norm_constant


def median_(stack):
    norm2fluxs = []
    for sed in stack:
        norm2fluxs.append(sed.norm2flux)

    norm_constant = numpy.median(norm2fluxs)
    return norm_constant


def value_(stack):
    return None


def _get_setattr(new_object, old_object):
    for key in old_object.__dict__:
        if key not in new_object.__dict__:
            setattr(new_object, key, old_object.__dict__[key])
