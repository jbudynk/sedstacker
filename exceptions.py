# Exceptions
# ----------
# InvalidSedType - if sed_type != 'sed' or 'spectrum'
# NonSupportedFileFormatError - for wrong fmt extensions (ascii, fits, votable), not enough columns
# PreExistingFile - raise if the file exists
# IOError - raise if the '/path/to/file' cannot be found
# UnrecognizedUnits - raise if input units are not included in astropy.units
# NonStandardColumnNames(KeyError) - raise if column names in SED File are not written in the SED File format
# NonExistantID - raise if someone tries to call a non-existant Sed/Spectrum from an AggregateSed object.
# 


class NoRedshiftError(Exception):
    def __init__(self):
        Exception.__init__(self, 'Redshift attribute z is set to None.')


class InvalidRedshiftError(Exception):
    def __init__(self, arg):
        if arg == 0:
            msg = 'Redshift must be float or int numeric type.'
        if arg == 1:
            msg = 'Redshift must be a positive number'
        Exception.__init__(self, msg)


class NegativeRedshiftRangeError(Exception):
    def __init__(self):
        Exception.__init__(self, 'Redshift must lie between 0 and 1100.')


class NoChangeInRedshiftError(Exception):
    def __init__(self):
        Exception.__init__(self, '')


class NotASegmentError(TypeError):
    def __init__(self):
        TypeError.__init__(self, 'AggregateSed object can only contain Segment class objects.')


class SegmentError(Exception):
    def __init__(self):
        Exception.__init__(self, '')


class OutsideRangeError(Exception):
    def __init__(self):
        Exception.__init__(self, 'The Segment does not fall within the specified spectral range for normalization')


class PreExistingFileError(Exception):
    def __init__(self, filename):
        Exception.__init__(self, '%s already exists. Aborting sed_write(). %s not written to file.' % (filename, filename))


class NonSupportedFileFormatError(Exception):
    def __init__(self, fmt):
        Exception.__init__(self, '%s is a non-supported file format.' % fmt)


class NonStandardColumnNamesError(Exception):
    def __init__(self):
        Exception.__init__(self, 'Column names must have at least one non-numeric character in the string name.\n [Ex 1: ID RA s16 ds16 (correct)]\n [Ex 2: ID RA 16 d16 (wrong)]')
