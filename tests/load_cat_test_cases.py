import numpy
import os.path
import os
import unittest
from sedstacker.sed import Sed, Spectrum, PhotometricPoint, AggregateSed, Segment
from sedstacker import io
from sedstacker.exceptions import NonStandardColumnNamesError, NonSupportedFileFormatError 


class TestLoadCat(unittest.TestCase):
	
    _rootdir = "/data/vao/staff/jbudynk/python_project/sedstacker/tests/test_data/"
    _column_map1 = {'s16': (16.0, 'micron', 'uJy', 'ds16'),
                    's24': (24.0, 'micron', 'uJy', 'ds24'),
                    's70': (70.0, 'micron', 'uJy', 'ds70')}

    def test_check_file_format(self):
        formats = ('ascii',)
	fmt = 'ascii'
	test_result = fmt in formats
	result = io.check_file_format('ascii')
	self.failUnless(result == test_result)

    def test_check_file_format_False(self):
        formats = ('ascii',)
        fmt = 'fits'
        result = io.check_file_format(fmt)
        self.failIf(result == True)


    def test_raises_NonSupportedFileFormat(self):
        self.assertRaises(NonSupportedFileFormatError, io.load_cat, self._rootdir+'3c273.csv', self._column_map1, fmt='csv')


    def testReadToSed(self):

        xarr = numpy.array([16.0, 24.0, 70.0])
        xunits = numpy.array(['micron','micron','micron'])
        yunits = numpy.array(['uJy','uJy','uJy'])

        sed = io.load_cat(self._rootdir+"gs_irs_sep9_one_source.dat",
                          self._column_map1)

        self.assertEqual(type(sed), Sed)
        self.assertAlmostEqual(sed[0].x, xarr[0])

	sed1 = Sed(x=xarr,
                   y=numpy.array([46.1, 104.0, -99.0]),
                   yerr=numpy.array([2.9, 6.2, -99.0]),
                   xunit=xunits,
                   yunit=yunits,
                   z=2.69)
        sed1.id = 'GS_IRS1'
        sed1.ra = '03:32:44.00'
        sed1.dec = '-27:46:35.0'

        self.assertEqual(sed.ra, sed1.ra)


    def test_raise_NonStandardColumnNames(self):

        self.assertRaises(NonStandardColumnNamesError, io.load_cat, self._rootdir+'load_cat_bad_column_names.dat', self._column_map1 )


    def testReadToAggregateSed_simple(self):

        aggsed1 = io.load_cat(self._rootdir+"gs_irs_sep9_small.dat", self._column_map1)

        xarr = numpy.array([16.0, 24.0, 70.0])
        xunits = numpy.array(['micron','micron','micron'])
        yunits = numpy.array(['uJy','uJy','uJy'])

        self.assertEqual(len(aggsed1),3)
        self.assertAlmostEqual(aggsed1[0][0].x, xarr[0])
        self.assertAlmostEqual(aggsed1[2][2].y, 2690.0)
        self.assert_(aggsed1.xunit[0][0] == xunits[0])
        
	sed1 = Sed(x=xarr,
                   y=numpy.array([46.1, 104.0, -99.0]),
                   yerr=numpy.array([2.9, 6.2, -99.0]),
                   xunit=xunits,
                   yunit=yunits,
                   z=2.69)
        sed1.id = 'GS_IRS1'
        sed1.ra = '03:32:44.00'
        sed1.dec = '-27:46:35.0'

        sed2 = Sed(x=xarr, 
                   y=numpy.array([163.1, 115.0, -99.0]),
                   yerr=numpy.array([2.8, 6.9, -99.0]),
                   xunit=xunits, yunit=yunits,
                   z=1.10)
        sed2.id = 'GS_IRS2'
        sed2.ra = '03:32:34.85'
        sed2.dec = '-27:46:40.0'

        sed3 = Sed(x=xarr,
                   y=numpy.array([2417.4, 3560.0, 2690.0]),
                   yerr=numpy.array([53.3, 36.0, 500.9]),
                   xunit=xunits,
                   yunit=yunits,
                   z=0.55)
        sed3.id = 'GS_IRS3'
        sed3.ra = '03:32:08.66'
        sed3.dec = '-27:47:34.4'

        aggsed2 = AggregateSed([sed1, sed2, sed3])
        
        self.failUnless(aggsed1[0][1].x == aggsed2[0][1].x)
        self.assertEqual(aggsed1[0].ra, sed1.ra)
        self.assertEqual(aggsed1[2].id, sed3.id)
        self.assertEqual(hasattr(aggsed1[1], 'z'), True)
        self.assertEqual(hasattr(aggsed1[1], 's24'), True)
        self.assertEqual(aggsed1[1][0].xunit, 'micron')


    def testReadToAggregateSed_complex(self):

        column_map = {"ucfht":(3823.0,"AA","mag","errU"),
                      "Bsub":(4459.7,"AA","mag","errB"),
                      "Vsub":(5483.8,"AA","mag","errV"),
                      "gsub":(4779.6,"AA","mag","errg"),
                      "rsub":(6295.1,"AA","mag","errR"),
                      "isub":(7640.8,"AA","mag","errI"),
                      "zsub":(9036.9,"AA","mag","errz"),
                      "j":(12491.0,"AA","mag","errJ"),
                      "Ks":(21590.4,"AA","mag","errK"),
                      "irac3.6":(36000.0,"AA","mag","err3.6"),
                      "irac4.5":(45000.0,"AA","mag","err4.5"),
                      "irac5.8":(58000.0,"AA","mag","err5.8"),
                      "irac8.0":(80000.0,"AA","mag","err8.0"),
                      "mips24.0":(240000.0,"AA","mag","err24")
                      }
        aggsed = io.load_cat(self._rootdir+"phot-cat-mags.ascii", column_map)

        self.assertEqual(aggsed.z[0], 0.668)
        self.assertEqual(aggsed.z[2], 0.529)
        self.assertEqual(aggsed[0].z, 0.668)
        self.assertEqual(aggsed[0].id, 2051)


    def testIncludeNames(self):
        self.assert_(True, 'Not yet implemented')

    
    def test_null_values(self):
        aggsed = io.load_cat(self._rootdir+"gs_irs_sep9_small.dat", self._column_map1)
        self.assertEqual(len(aggsed[0]), 2)
        self.assertEqual(len(aggsed[1]), 3)
        self.assertEqual(len(aggsed[2]), 3)
        self.assert_(numpy.isnan(aggsed.yerr[1][2]))
        self.assert_(type(aggsed.yerr[2][2]) is numpy.float_)


    def test_no_flux_errors(self):
        column_map = {'s16': (16.0, 'micron', 'uJy', 'ds16'),
                      's24': (24.0, 'micron', 'uJy', None),
                      's70': (70.0, 'micron', 'uJy', 'ds70')}
        aggsed = io.load_cat(self._rootdir+"gs_irs_sep9_small.dat", column_map)
        self.assert_(numpy.isnan(aggsed.yerr[1][1]))
        self.failIf(numpy.isnan(aggsed.yerr[2][2]))


def main():
    unittest.main()

if __name__ == '__main__':
    main()
