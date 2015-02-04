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
