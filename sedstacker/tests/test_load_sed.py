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
from sedstacker import io
from numpy import array, nan, testing

test_directory = os.path.dirname(sedstacker.__file__)+"/tests/resources/test_load_sed/"

class TestLoadSed(unittest.TestCase):
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

    def test_load_spectrum_three_cols(self):

        spectrum = io.load_sed(test_directory+'case1.dat',
                               sed_type='spectrum')

        self.assertEqual(spectrum.x[0], 814.020019531)
        self.assertEqual(spectrum.y[3], -1.21521716301e-14)

        sed = io.load_sed(test_directory+'case1.dat', sed_type='sed')

        self.assertEqual(sed[0].x, 814.020019531)
        self.assertEqual(sed[3].y, -1.21521716301e-14)

    
    def test_load_spectrum_three_cols_no_col_names(self):
        
        spectrum = io.load_sed(test_directory+'case2_3columnsNoNames.dat',
                               sed_type='spectrum')

        #testing.assert_array_equal(sed.x, control_x)
        #testing.assert_array_equal(sed.y, control_y)
        #testing.assert_array_equal(sed.yerr, control_yerr)
        #testing.assert_array_equal(sed.xunit, control_xunit)
        #testing.assert_array_equal(sed.yunit, control_yunit)

        pass

    def test_load_sed_no_yerr_no_col_names(self):

        sed = io.load_sed(test_directory+'case4_2columnsNoNames.dat',
                          sed_type='sed')

        control_x = array([814.020019531, 814.538879394, 815.057678223,
                           815.576538086, 816.095336914])
        control_y = array([-1.21828545516e-14, -1.21726630511e-14,
                            -1.21624715507e-14, -1.21521716301e-14,
                            -1.21419801297e-14])
        control_yerr = array([nan, nan, nan, nan, nan])
        control_xunit = array(['AA','AA','AA','AA','AA'])
        control_yunit = array(['erg/s/cm**2/AA','erg/s/cm**2/AA',
                               'erg/s/cm**2/AA','erg/s/cm**2/AA',
                               'erg/s/cm**2/AA'])
        
        testing.assert_array_almost_equal(sed.x, control_x)
        testing.assert_array_almost_equal(sed.y, control_y)
        testing.assert_array_almost_equal(sed.yerr, control_yerr)
        testing.assert_array_equal(sed.xunit, control_xunit)
        testing.assert_array_equal(sed.yunit, control_yunit)


if __name__ == '__main__':
    unittest.main()
        
