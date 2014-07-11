import numpy
import types
import warnings
import logging
import types
from sedstacker.exceptions import SegmentError

logger=logging.getLogger(__name__)
formatter=logging.Formatter('%(levelname)s:%(message)s')
hndlr=logging.StreamHandler()
hndlr.setFormatter(formatter)
logger.addHandler(hndlr)

def binup(y, x, xarr, statistic, binsize, fill, yerr, logbin = False):

    builtin_statistics = {'avg':avg_bin, 'wavg':wavg_bin, 'sum':sum_bin}

    if isinstance(statistic, types.StringType):
        if statistic not in builtin_statistics.keys():
            raise ValueError('Unknown built-in statistic. Choices are:\n'+
                             '"sum"  - sum the values in each bin\n'+
                             '"avg"  - (DEFAULT) average the values in each'+
                             ' bin\n'+
                             '"wavg" - compute the weighted average of the'+
                             ' values in each bin\n'+
                             'Otherwise, input a user-defined function: \n\t'+
                             'y_combined, yerr_combined, number_of_combined_ys'+
                             ' = func(y_bin, yerr_bin, count)')
        else:
            statistic = builtin_statistics[statistic]
    else:
        if not isinstance(statistic, types.FunctionType):
            raise ValueError('Unknown built-in statistic. Choices are:\n'+
                             '"sum"  - sum the values in each bin\n'+
                             '"avg"  - (DEFAULT) average the values in each'+
                             ' bin\n'+
                             '"wavg" - compute the weighted average of the'+
                             ' values in each bin\n'+
                             'Otherwise, input a user-defined function: \n\t'+
                             'y_combined, yerr_combined, number_of_combined_ys'+
                             ' = func(y_bin, yerr_bin, count)')
        else:
            statistic = statistic
        
    if fill not in ('remove','fill'):
        raise ValueError('kwarg fill must be \'fill\' or \'remove\'.')

    x, xarr, y, yerr, nx, xbin, yarr, outerr, count, skipit = \
        setup_binup_arrays(y, x, xarr, binsize, yerr, logbin = logbin)

    nans_present = False
    if any(numpy.isnan(yerr)):
        nans_present = True
        logger.info('NaNs found in flux errors. \'outerr\' array is variance of flux per bin.')
        if statistic==wavg_bin:
            statistic = avg_bin
            logger.warning('NaNs found in flux errors. Using \'avg\' statistic instead of \'wavg.\'')

    warnings.simplefilter("ignore", UserWarning)
    total=0

    for i in range(nx):
        lim = xarr[i] + xbin
        cut = numpy.less(x, lim)
        y_bin = y[cut]
        yerr_bin = yerr[cut]

        cut = numpy.invert(cut)
        x = x[cut]
        y = y[cut]
        yerr = yerr[cut]

        count[i] = len(y_bin)

        if count[i] >= 1:
            yarr[i], outerr[i], count[i] = statistic(y_bin, yerr_bin, nans_present)
        else:
            skipit[i] = 0

    if logbin:
        xarr = 10**xarr

    mask = numpy.ma.make_mask(skipit)
    if fill == 'fill':
        xarr = xarr
        yarr = fill_fill(mask, yarr)
        outerr = fill_fill(mask, outerr)
        counts = count
    else:
        xarr = fill_remove(mask, xarr)
        yarr = fill_remove(mask, yarr)
        outerr = fill_remove(mask, outerr)
        counts = fill_remove(mask, count)

    return yarr, xarr, outerr, counts   


def wavg_bin(y_bin, yerr_bin, nans_present):

    if nans_present:
        yarr = numpy.mean(y_bin)
        outerr = numpy.std(y_bin)
    else:
        weights = 1.0/yerr_bin**2
        yarr = numpy.average(y_bin, weights=weights)
        outerr = numpy.sqrt((yerr_bin**2).sum())

    count = len(y_bin)
    
    return yarr, outerr, count


def avg_bin(y_bin, yerr_bin, nans_present):

    if nans_present:
        outerr = numpy.std(y_bin)
    else:
        outerr = numpy.sqrt((yerr_bin**2).sum())
    yarr = numpy.mean(y_bin)
    count = len(y_bin)

    return yarr, outerr, count


def sum_bin(y_bin, yerr_bin, nans_present):

    if nans_present:
        outerr = numpy.std(y_bin)
    else:
        outerr = numpy.sqrt((yerr_bin**2).sum())
    yarr = y_bin.sum()
    count = len(y_bin)
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


def setup_binup_arrays(y, x, xarr, binsize, yerr, logbin=False):

    xbin = binsize/2.0
    nx = len(xarr)

    y = numpy.array(y)
    yerr = numpy.array(yerr)

    if logbin:
        x = numpy.log10(x)
        xarr = numpy.log10(xarr)
    else:
        x = numpy.array(x)

    # to be binned y-values and y-error values
    yarr = xarr*0
    outerr = xarr*0
    count = numpy.int_(xarr*0)

    skipit = numpy.ones(len(yarr))

    return x, xarr, y, yerr, nx, xbin, yarr, outerr, count, skipit


def _sorts(x, y, yerr):
    points = []
    for i, point in enumerate(x):
        points.append([x[i], y[i], yerr[i]])
    points = zip(*sorted(points))
    x = numpy.array(points[0])
    y = numpy.array(points[1])
    yerr = numpy.ma.masked_invalid(points[2])

    return x, y, yerr 


def smooth(arr, smooth_binsize):
    window = numpy.ones(int(smooth_binsize))/float(smooth_binsize)
    return numpy.convolve(arr, window, 'same')


def big_spec(arr,binsize,log):

    if log:
        arr_min = numpy.floor(min(numpy.log10(arr)))
        arr_max = numpy.ceil(max(numpy.log10(arr)))
        return 10**(numpy.arange(arr_min, arr_max+binsize, binsize))
    else:
        arr_min = numpy.floor(min(arr))
        arr_max = numpy.ceil(max(arr))
        return numpy.arange(arr_min, arr_max+binsize, binsize)


def interpolate(yin, xin, xout, method='linear'): 
    """ 
    Interpolate the curve defined by (xin, yin) at points xout. The array 
    xin must be monotonically increasing. The output has the same data type as 
    the input yin.
 
    Parameters
    ----------
    yin : array-like
        y values of input curve 
    xin : array-like
        x values of input curve 
    xout : array-like
        x values of output interpolated curve 
    method : str
        interpolation method ('linear' | 'nearest'). Default is 'linear' 
 
    Returns
    -------
    numpy array with interpolated curve 

    Notes
    -----
    Code adopted from Astropython.org [1]_.
    
    References
    ----------
    ..[1] http://www.astropython.org/snippet/2010/11/Interpolation-without-SciPy

    """  

    lenxin = len(xin)  

    i1 = numpy.searchsorted(xin, xout)
    i1[ i1==0 ] = 1
    i1[ i1==lenxin ] = lenxin-1

    x0 = xin[i1-1]
    x1 = xin[i1]
    y0 = yin[i1-1]
    y1 = yin[i1]
  
    if method == 'linear':
        return (xout - x0) / (x1 - x0) * (y1 - y0) + y0
    elif method == 'nearest':
        return numpy.where(numpy.abs(xout - x0) < numpy.abs(xout - x1), y0, y1)
    else:
        raise ValueError('Invalid interpolation method: %s' % method)


def fast_nearest_interp(xi, x, y):
    """Assumes that x is monotonically increasing!!."""
    # Shift x points to centers
    spacing = numpy.diff(x) / 2
    x = x + numpy.hstack([spacing, spacing[-1]])
    # Append the last point in y twice for ease of use
    y = numpy.hstack([y, y[-1]])
    return y[numpy.searchsorted(x, xi)]
