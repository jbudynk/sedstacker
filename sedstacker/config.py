#
from numpy import float_, int_
from types import FloatType, IntType
# list of acceptable null values
# when reading data from file
NONE_VALS = (None, 'None', 'none', 'null', 'nan', 'NaN', -99., -99)
NUMERIC_TYPES = (FloatType, IntType, float_, int_)

