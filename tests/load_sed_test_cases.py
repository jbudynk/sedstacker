#
import unittest
import os
import os.path
from sedstacker import io

class TestSedFileInput(unittest.TestCase):
    '''
    Test the SED File reader functions, load_sed, check_fmt, and _read_ascii.
    
    Cases:

    1. SED File with three columns named x, y and y_err (succeed)
    2. SED File with three columns, no column names (succeed with NoName warning / NonStandardColumnNames)
    3. SED File with two columns named x and y (succeed with NoYErr warning)
    4. SED File with two columns, no column names (succeed with NoName and NoYErr warnings / NonStandardColumnNames)
    5. SED File with three columns named y, y_err and x (succeed)
    6. SED File with three columns named x, y and z (succeed with NoYErr warning)
    7. SED File with three columns named 123, y and y_err (fail - raise AstroPy exception - invalid column name)
    8. SED File with three columns, named correctly, some cells with 'null' values (succeed)
    9. SED File with three columns named x, y and y_err, but one of the columns does not have the same number of rows as the other columns (fail - raise AstroPy exception - not all rows/columns have same length)
    10. SED File with more than 3 columns, with three columns named x, y and y_err. (succeed)
        10.1 Save other columns as attributes to Sed/Spectrum object (succeed)
    11. SED File with one column (fail - raise NonSupportedFileFormatError - not enough columns)
    12. Read SED File in with fmt='fits' (fail - raise NonSupportedFileFormatError)
    13. Read SED File with sed_type = 1, 'not a format','sd' (fail - raise InvalidSedTypeError)
    14. Read file with numerous comments before column names (succeed)

    . Read SED File with 11 columns and sed_type = 'sed' (fail - load_sed() cannot read more than 10 columns into a Sed object)
    . Read IPAC SED File


    '''

    rootdir = "/data/vao/staff/jbudynk/python_project/sedstacker/tests/test_load_sed/"

    def testThreeColsWithNames(self):

        spectrum = io.load_sed(self.rootdir+'case1.dat', sed_type='spectrum')

        self.assertEqual(spectrum.x[0], 814.020019531)
        self.assertEqual(spectrum.y[3], -1.21521716301e-14)

        sed = io.load_sed(self.rootdir+'case1.dat', sed_type='sed')

        self.assertEqual(sed[0].x, 814.020019531)
        self.assertEqual(sed[3].y, -1.21521716301e-14)

    
    def testThreeColsWithoutNames(self):
        
        spectrum = io.load_sed(self.rootdir+'case2_3columnsNoNames.dat', sed_type='spectrum')
        pass


if __name__ == '__main__':
    unittest.main()
        
