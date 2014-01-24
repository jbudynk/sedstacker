import numpy
import sedstacker
from sedstacker.sed import Sed, Spectrum, AggregateSed, create_from_points, PhotometricPoint
from sedstacker.exceptions import PreExistingFileError
import unittest
import os.path
import os
import astropy.io.ascii

test_directory = os.path.dirname(sedstacker.__file__)+"/tests/resources/write/"

class TestWriteSed(unittest.TestCase):

    x = numpy.linspace(1000,10000,num=100)
    y = numpy.linspace(1000,10000,num=100)*1e-10
    yerr = y*0.01

    def setUp(self):
        filename = test_directory+'test_PrexistingFile.txt'
        f = open(filename, 'w')
        f.write('oh hey there')
        f.close()


    def tearDown(self):
        os.system('cp '+test_directory+'*.* '+test_directory+'copy/')
        os.system('rm -f '+test_directory+'*.*')


    def test_sed_write_file_exists(self):

        sed = Sed(x=self.x,y=self.y,yerr=self.yerr)

        filename = test_directory+'sed_file_exists.dat'
        sed.write(filename)
        self.assertEqual(os.path.exists(filename), True)


    def test_sed_write_with_counts(self):

        sed = Sed(x=self.x,y=self.y,yerr=self.yerr)
        sed.counts = numpy.ones(numpy.array(sed._toarray()[0]).size, dtype=numpy.int_)
        filename = test_directory+'sed_counts.dat'
        sed.write(filename)
        self.assertEqual(os.path.exists(filename), True)
        data = astropy.io.ascii.read(filename)
        self.assertEqual(data['counts'][3],1.0)


    def test_spectrum_write_file_exists(self):
        spectrum = Spectrum(x=self.x, y=self.y, yerr=self.yerr)
        filename = test_directory+'spectrum_file_exists.dat'
        spectrum.write(filename)
        self.assertEqual(os.path.exists(filename), True)


    def test_aggsed_write_file_exists(self):
        sed = Sed(x=self.x,y=self.y,yerr=self.yerr)
        spectrum = Spectrum(x=self.x, y=self.y, yerr=self.yerr)

        aggsed = AggregateSed([sed, spectrum])
        filename = test_directory+'aggsed_file_exists.dat'
        aggsed.write(filename)
        self.assertEqual(os.path.exists(filename), True)


    def test_aggsed_counts1(self):
        sed = Sed(x=self.x,y=self.y,yerr=self.yerr)
        sed.counts = numpy.ones(numpy.array(sed._toarray()[0]).size, dtype=numpy.int_)
        spectrum = Spectrum(x=self.x, y=self.y, yerr=self.yerr)

        aggsed = AggregateSed([sed, spectrum])

        filename = test_directory+'aggsed_counts1.dat'
        aggsed.write(filename)
        self.assert_(os.path.exists(filename))
        data = astropy.io.ascii.read(filename)
        self.assertEqual(all(counts == numpy.nan for counts in data['counts']), False)


    def test_aggsed_no_counts2(self):
        sed = Sed(x=self.x,y=self.y,yerr=self.yerr)
        spectrum = Spectrum(x=self.x, y=self.y, yerr=self.yerr)

        aggsed = AggregateSed([sed, spectrum])

        filename = test_directory+'aggsed_counts2.dat'
        aggsed.write(filename)
        data = astropy.io.ascii.read(filename)
        self.assertEqual(data.colnames, ['x','y','y_err'])


    def test_raise_PreExistingFile(self):
        sed = Sed(x=self.x,y=self.y,yerr=self.yerr)
        spectrum = Spectrum(x=self.x, y=self.y, yerr=self.yerr)
        aggsed = AggregateSed([sed, spectrum])
        
        filename = test_directory+'test_PrexistingFile.txt'
        self.assertRaises(PreExistingFileError, aggsed.write, filename)


if __name__ == '__main__':
    unittest.main()
