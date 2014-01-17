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
stack_ = stack_spectra.toarray()

stack_spectra.write(test_directory+'stacked_spectra_maskcc10.dat')

stack_spectra = load_sed(test_directory+'stacked_spectra_maskcc10.dat')

end = time.clock()

plt.plot(stack_[0], stack_[1])
plt.show()

print ''
print 'time it took to do all of this: %.3g sec' % (end-start)
