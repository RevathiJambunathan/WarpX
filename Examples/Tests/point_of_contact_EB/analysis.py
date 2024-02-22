#!/usr/bin/env python

"""
This script tests the coordinates of the point of contact of an electron hitting a sphere in 3D.
It compares the numerical results with the analytical solutions.
The sphere is centered on O and has a radius of 0.2 (EB)
The electron is initially at: (-0.25,0,0) and moves with a normalized momentum: (1,0.5,0)
An input file PICMI_inputs_3d.py is used.
"""
import os
import sys

import numpy as np
from openpmd_viewer import OpenPMDTimeSeries
import yt

yt.funcs.mylog.setLevel(0)
sys.path.insert(1, '../../../../warpx/Regression/Checksum/')
import checksumAPI

# Open plotfile specified in command line
filename = sys.argv[1]
test_name = os.path.split(os.getcwd())[1]
checksumAPI.evaluate_checksum(test_name, filename, output_format='openpmd')

ts_scraping = OpenPMDTimeSeries('./diags/diag2/particles_at_eb/')

it=ts_scraping.iterations
step_scraped, time_scraped, x, y, z=ts_scraping.get_particle( ['stepScraped','timeScraped','x','y','z'], species='electron', iteration=it )
time_scraped_reduced=time_scraped[0]*1e10

# Analytical results calculated
x_analytic=-0.1983
y_analytic=0.02584
z_analytic=0.0000

#result obtained by analysis of simulations
step=3
time_reduced=3.58

print('NUMERICAL coordinates of the point of contact:')
print('step_scraped=%d, time_stamp=%5.4f e-10, x=%5.4f, y=%5.4f, z=%5.4f' % (step_scraped[0],time_reduced,x[0], y[0], z[0]))
print('\n')
print('ANALYTICAL coordinates of the point of contact:')
print('step_scraped=%d, time_stamp=%5.4f e-10, x=%5.4f, y=%5.4f, z=%5.4f' % (step, time_reduced, x_analytic, y_analytic, z_analytic))

tolerance=0.001
tolerance_t=0.003
print("tolerance = "+ str(tolerance *100) + '%')
print("tolerance for the time = "+ str(tolerance_t *100) + '%')

diff_step=np.abs((step_scraped[0]-step)/step)
diff_time=np.abs((time_scraped_reduced-time_reduced)/time_reduced)
diff_x=np.abs((x[0]-x_analytic)/x_analytic)
diff_y=np.abs((y[0]-y_analytic)/y_analytic)

print("percentage error for x = %5.4f %%" %(diff_x *100))
print("percentage error for y = %5.4f %%" %(diff_y *100))

assert (diff_x < tolerance) and (diff_y < tolerance) and (np.abs(z[0]) < 1e-8) and (diff_step < 1e-8) and (diff_time < tolerance_t) , 'Test point_of_contact did not pass'