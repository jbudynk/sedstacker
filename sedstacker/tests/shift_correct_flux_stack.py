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

import time
import sedstacker
from sedstacker.sed import Spectrum, Sed, AggregateSed, stack
from sedstacker.io import load_sed
from matplotlib import pyplot as plt
import numpy as np
import logging
import os

start = time.clock()
logger_io=logging.getLogger('sedstacker.io')
logger_io.setLevel(logging.ERROR)
logger_sed=logging.getLogger('sedstacker.sed')
logger_sed.setLevel(logging.ERROR)

test_directory = os.path.dirname(sedstacker.__file__)+"/tests/resources/spectra/"

os.system('rm '+test_directory+'stacked_spectra_maskcc10.dat')
files = os.listdir(test_directory)

specs = []
counter = 0
while counter < 14:
    spec = load_sed(test_directory+files[counter], sed_type="spectrum")
    specs.append(spec)
    counter += 1

#specs = [load_sed(test_directory+f, sed_type="spectrum") for f in files]
for spec in specs:
    spec.z = np.random.random_sample()

aggsed = AggregateSed(specs)

norm_aggsed = aggsed.normalize_at_point(3500.0, 1.0)

stack_spectra = stack(norm_aggsed, 10.0, 'avg', fill='fill')

stack_spectra.write(test_directory+'stacked_spectra_maskcc10.dat')

stack_spectra = load_sed(test_directory+'stacked_spectra_maskcc10.dat')

end = time.clock()

plt.plot(stack_.x, stack_.y)
plt.show()

print ''
print 'time it took to do all of this: %.3g sec' % (end-start)
