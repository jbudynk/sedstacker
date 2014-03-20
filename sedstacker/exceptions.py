# 

class NoRedshiftError(Exception):
    """
    Raise if the redshift attribute `z` of a Segment is None.

    Parameters
    ----------
    msg : str (optional)
        Error message. Default is *Redshift attribute z is set to None.*

    """

    msg = 'Redshift attribute z is set to None.'

    def __init__(self, msg=msg):
        Exception.__init__(self, self.msg)


class InvalidRedshiftError(Exception):
    """
    Case 0: Raise if either the target redshift `z0` or the redshift attribute
    `z` isn't a numeric type.

    Case 1: Raise if either the target redshift `z0` or the redshift attribute
    `z` is negative. Redshifts must be greater than or equal to 0.

    Case num: Raise if you want to change the output error message.

    Parameters
    ----------
    error : int
        Flag to specify which message to write.

        0 - [default] if "Case 0" applies. Message: *Redshift must be float or int numeric type.*

        1 - if "Case 1" applies. Message: *Redshift cannot be negative.*

        num - where *num* can be any integer other than 0 or 1. Let's you provide your own error message.

    """
    
    msg = ''
    
    def __init__(self, error, msg=msg):
        if error == 0:
            msg = 'Redshift must be float or int numeric type.'
        elif error == 1:
            msg = 'Redshift cannot be negative.'
        else:
            msg = 'watch out! didn\'t add a custom error message'
        Exception.__init__(self, msg)


class NotASegmentError(TypeError):
    """
    Raise if a function, method, or class argument that accepts only Segment 
    objects recieves a an argument that is not a Segment.

    Parameters
    ----------
    msg : str (optional)
        Error message. Default is *AggregateSed object can only contain Segment
        class objects.*
    
    """

    msg = 'AggregateSed object can only contain Segment class objects.'

    def __init__(self, msg=msg):
        TypeError.__init__(self, msg)


class SegmentError(Exception):
    """
    Raise if an error related to Segment creation/manipulation occurs. For 
    example, if len(x) != len(y), and one tries to instantiate a Sed with

    >>> sed = sedstacker.Sed(x=x,y=y)

    a SegmentError is raised.

    Parameters
    ----------
    msg : str
        The message to output to screen if SegmentError is raised.

    """

    def __init__(self, msg):
        Exception.__init__(self, msg)


class OutsideRangeError(Exception):
    """
    Raise if the user-defined spectral range over which to normalize a Segment
    does not fall within the spectral range of the Segment, `x`.

    Parameters
    ----------
    msg : str (optional)
        Error message. Default is *The Segment does not fall within the 
        specified spectral range for normalization.*

    """

    msg = ('The Segment does not fall within the specified spectral range for '+ 
           'normalization')

    def __init__(self, msg=msg):
        Exception.__init__(self, msg)


class PreExistingFileError(Exception):
    """
    When writing data to file, raise if the input file name already exists.

    Parameters
    ----------
    filename : str
        Name of the file.
    msg : str (optional)
        Error message. Default is *filename already exists. Aborting write(). 
        filename not written to file.*

    """

    msg = ''

    def __init__(self, filename, msg=msg):
        if msg == '':
            msg = ('%s already exists. ' % (filename) + 
                   'Aborting write(). %s' % (filename) + 
                   ' not written to file.')
        Exception.__init__(self, msg)


class NonSupportedFileFormatError(Exception):
    """
    Raise if user specifies an unsupported file format.

    Parameters
    ----------
    fmt : str
        String of file format.
    msg : str (optional)
        Error message. Default is *fmt is a non-supported file format.*

    """

    msg = ''

    def __init__(self, fmt, msg=msg):
        if msg == '':
            '%s is a non-supported file format.' % fmt
        Exception.__init__(self, msg)


class NonStandardColumnNamesError(Exception):
    """
    Raise if column names in a Photometry Catalog are in an incorrect format
    for astropy.io.ascii.read() (extrnal dependency).
    
    """
    def __init__(self):
        Exception.__init__(self, 'Column names must begin with an alphabetic' + 
                           ' character in the string name.\n ' + 
                           '[Ex 1: ID RA s16 ds16 (correct)]\n [Ex 2: ID RA ' +
                           '16 d16 (wrong)]')
