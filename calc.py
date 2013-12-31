import numpy
import math
import types
from sedstacker.exceptions import SegmentError

# to get times:
# 
# import time
#
# start = time.clock()
# do_some_stuff()
# end = time.clock()
# print ' %.3g s' % (end-start)


def binup(y, x, xarr, statistic, binsize, fill, yerr, logbin = False):

    #numpy.seterr(all='ignore')

    builtin_statistics = {'avg':avg_bin, 'wavg':wavg_bin, 'sum':sum_bin}

    if isinstance(statistic, types.StringType):
        if statistic not in builtin_statistics.keys():
            raise ValueError('Unknown built-in statistic. Choices are:\n'+
                             '"sum"  - sum the values in each bin\n'+
                             '"avg"  - (DEFAULT) average the values in each bin\n'+
                             '"wavg" - compute the weighted average of the values in each bin\n'+
                             'Otherwise, input a user-defined function: \n\t'+
                             'y_combined, yerr_combined, number_of_combined_ys = func(y_bin, yerr_bin, count)')

        else:
            statistic = builtin_statistics[statistic]

    else:
        if not isinstance(statistic, types.FunctionType):
            raise ValueError('Unknown built-in statistic. Choices are:\n'+
                             '"sum"  - sum the values in each bin\n'+
                             '"avg"  - average the values in each bin\n'+
                             '"wavg" - compute the weighted average of the values in each bin\n'+
                             'Otherwise, input a user-defined function: \n\t'+
                             'y_combined, yerr_combined, number_of_combined_ys = func(y_bin, yerr_bin, count)')

        else:
            statistic = statistic
        
    if fill not in ('remove','fill'):
        raise ValueError('kwarg fill must be \'fill\' or \'remove\'.')

    x, xarr, y, m_yerr, nx, xbin, yarr, outerr, count, skipit = setup_binup_arrays(y, x, xarr, binsize, yerr, logbin = logbin)

    for i in range(nx):
        high_lim = xarr[i] + xbin
        low_lim = xarr[i] - xbin
        # the next 2 lines are very inefficient - for 6 sources, takes ~12.5 secs
        #y_bin = y[(x >= low_lim) & (x <= high_lim)]
        #yerr_bin = m_yerr[(x >= low_lim) & (x<= high_lim)]

        # This method takes ~10.5 seconds for 6 sources
        low_cut = numpy.greater_equal(x, low_lim)
        high_cut = numpy.less_equal(x, high_lim)
        total_cut = numpy.logical_and(low_cut, high_cut)
        y_bin = y[total_cut]
        yerr_bin = m_yerr[total_cut]
        count[i] = len(y_bin)

        if count[i] >= 1:
            yarr[i], outerr[i], count[i] = statistic(y_bin, yerr_bin, count[i])

        else:
            skipit[i] = 0

    mask = numpy.ma.make_mask(skipit)

    if fill == 'fill':
        out_xarr = xarr
        out_yarr = fill_fill(mask, yarr)
        out_err = fill_fill(mask, outerr)
        counts = count
    else:
        out_xarr = fill_remove(mask, xarr)
        out_yarr = fill_remove(mask, yarr)
        out_err = fill_remove(mask, outerr)
        counts = fill_remove(mask, count)

    return out_yarr, out_xarr, out_err, counts


def wavg_bin(y_bin, yerr_bin, count):

    #if len(yerr_bin) == count:
    weights = 1.0/yerr_bin**2
    yarr = numpy.average(y_bin, weights=weights)
    outerr = math.sqrt((yerr_bin**2).sum())
    # outerr[i] = numpy.std(y_bin)) #take either the stddev of y, or sum in quadrature of the errors
    # If any NaN's exist in yerr_bin, then their corresponding fluxes
    # will not be taken into account for the weighted average.
    # The number of flux counts would be less than the number
    # of points in xbin.
    count = len(numpy.where(yerr_bin.mask == False)[0])

    #else:
    #    yarr = numpy.average(y_bin)
    #    outerr = math.sqrt((yerr_bin**2).sum())
    #    print 'len(yerr) does not match len(y). Computing average instead.'
    
    return yarr, outerr, count


def avg_bin(y_bin, yerr_bin, count):

    yarr = numpy.average(y_bin)
    # outerr[i] = numpy.std(y_bin)
    outerr = math.sqrt((yerr_bin**2).sum())

    return yarr, outerr, count


def sum_bin(y_bin, yerr_bin, count):

    yarr = y_bin.sum()
    # outerr = numpy.std(y_bin)
    outerr = math.sqrt((yerr_bin**2).sum())

    return yarr, outerr, count


def fill_fill(mask, arr):
    new_array = numpy.zeros(len(arr))
    for i, val in enumerate(mask):
        if val:
            new_array[i] = arr[i]
        else:
            new_array[i] = numpy.nan
    
    return new_array


def fill_remove(mask, arr):
    new_array = []
    for i, val in enumerate(mask):
        if val:
            new_array.append(arr[i])
    
    return numpy.array(new_array)


def smooth(arr, smooth_binsize):
    window = numpy.ones(int(smooth_binsize))/float(smooth_binsize)
    return numpy.convolve(arr, window, 'same')


def setup_binup_arrays(y, x, xarr, binsize, yerr, logbin = False):

    if logbin:
        x = numpy.log10(numpy.array(x))
        xarr = numpy.log10(numpy.array(xarr))
    else:
        x = numpy.array(x)
        xarr = numpy.array(xarr)

    y = numpy.array(y)
    m_yerr = numpy.ma.masked_array(yerr,numpy.isnan(yerr))

    nx = len(xarr)
    xbin = binsize/2.0

    #returns the binned y-values and y-error values
    yarr = xarr*0
    outerr = xarr*0
    count = numpy.int_(xarr*0)

    skipit = numpy.ones(len(yarr))

    return x, xarr, y, m_yerr, nx, xbin, yarr, outerr, count, skipit


def big_spec(arr,binsize,log):
    if log:
        arr_min = math.floor(min(arr))
        arr_max = math.ceil(max(arr))
        return numpy.logspace(math.log10(arr_min), math.log10(arr_max), num=(arr_max-arr_min)/binsize)
    else:
        arr_min = math.floor(min(arr))
        arr_max = math.ceil(max(arr))
        return numpy.arange(arr_min, arr_max+binsize, binsize)
