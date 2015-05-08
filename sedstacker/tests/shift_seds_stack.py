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

import numpy
import os
import logging
import sedstacker
import unittest
from sedstacker import io
from sedstacker.sed import AggregateSed, Sed, stack
import matplotlib.pyplot as plt
import datetime

logging.basicConfig(level=logging.INFO)

rootdir = os.path.dirname(sedstacker.__file__)
filename = rootdir+"/tests/resources/phot-cat-mags.ascii"

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

aggsed = io.load_cat(filename, column_map)
#plt.plot(aggsed.x,aggsed.y,'o')
#plt.show()

# convert from ABmag to flux density, Jansky
c = 2.5 / numpy.log(10.0)
for i, sed in enumerate(aggsed):
    for j, point in enumerate(sed):
        point.y = 10**23 * 10**(-(point.y+48.6) / 2.5)
        point.yerr = point.yerr * aggsed.y[i][j] / c
        aggsed.y[i][j] = point.y
        aggsed.yerr[i][j] = point.yerr

#plt.plot(aggsed.x,aggsed.y,'o')
#plt.show()

restframe_aggsed = aggsed.shift(0)

#plt.loglog(restframe_aggsed.x[0],restframe_aggsed.y[0],'o',restframe_aggsed.x[1],restframe_aggsed.y[1],'o',restframe_aggsed.x[2],restframe_aggsed.y[2],'o',restframe_aggsed.x[3],restframe_aggsed.y[3],'o',restframe_aggsed.x[4],restframe_aggsed.y[4],'o',restframe_aggsed.x[5],restframe_aggsed.y[5],'o')
#plt.show()

norm_point_aggsed = restframe_aggsed.normalize_at_point(5000.0, 1e-5, norm_operator=1)
print ''
norm_int_aggsed = restframe_aggsed.normalize_by_int()

plt.plot(restframe_aggsed.x[1],restframe_aggsed.y[1],'ko', norm_point_aggsed.x[1],norm_point_aggsed.y[1],'go', norm_int_aggsed.x[1],norm_int_aggsed.y[1],'ro')
plt.legend(('restframe','norm_at_point','norm_by_int'), fontsize='x-small', loc=2)
plt.show()

start = datetime.datetime.now()
stack_i = stack(norm_int_aggsed, 0.1, 'wavg', fill='remove', logbin=True)
stack_p = stack(norm_point_aggsed, 0.1, 'wavg', fill='remove', logbin=True)
stack_rf = stack(restframe_aggsed, 0.1, 'wavg', fill='remove', logbin=True)
end = datetime.datetime.now()

print(end - start)

plt.loglog(stack_rf.x,stack_rf.y,'ko',stack_p.x,stack_p.y,'go',stack_i.x,stack_i.y,'ro')
plt.legend(('restframe','norm_at_point','norm_by_int'),fontsize='x-small', loc=2)
plt.show()


