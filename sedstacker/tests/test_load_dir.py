import unittest
import os
import logging

import sedstacker
from sedstacker.io import load_dir

test_directory = os.path.dirname(sedstacker.__file__)+"/tests/resources/"

logger=logging.getLogger('sedstacker.io')
logger.setLevel(logging.ERROR)

class TestLoadDir(unittest.TestCase):
    
    def test_load_dir(self):
        directory = test_directory+"spectra/"
        aggsed = load_dir(directory)
        self.assertEqual(len(aggsed), len(os.listdir(directory)))


if __name__ == '__main__':
    unittest.main()
