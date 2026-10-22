#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ERT modeling and inversion in 2D
================================

This code uses the open-source library: <br/> pyGIMLi (www.pygimli.org) and PyBERT (https://pypi.org/project/PyBERT/) to create mesh, perform forward modeling and run the inversion. 

A number of python libraries such as matplotlib, numpy, pandas are also used. 
The file geo05.yml is a list of describing python environment where all necessary libraries are installed. 
The yml file can be used to re-create the proper environment with conda. 

**To run, this code requires the companion file:**<br>
Init_ERT2D.py

S. Nielsen, March 2026
"""

# [notebook-magic removed] %matplotlib ipympl
from matplotlib import pyplot as plt
from Init_ERT2D_3_4 import *
import pygimli as pg
from pygimli.physics import ert

#%%
# READ DATA EXPORTED FROM ABEM TERRAMETER <Export task as txt>
# You need to edit the name of input file here to match your input data file.

#infile = 'Grad-064E-01S-2026-03-04-122339.txt'
infile = 'Stef_Gradient_64_1-2026-03-24-152000.txt'
#infile='Stef_GradientXL_64_1-2026-03-26-171817.txt'
#infile='Stef_GradientXL_64_2-2026-03-26-171904.txt'
#infile='Stef_GradientXL_64_4-2026-03-26-172128.txt'
#infile='Stef_Gradient_64_3-2026-03-25-200111.txt' --> faulty!
#infile = 'Stef_Gradient_64_3-2026-03-25-130728.txt' -->faulty!
#infile = 'Continuity_1-2026-03-24-152102.txt' -->faulty!
data = ert.load(infile)
Nel=data.sensorCount()
print(data)

#%%
'''
----------------------------------------------------------------------------
IF THERE IS TOPOGRAPHY (if negligible topography: just skip this section)
----------------------------------------------------------------------------

Input topography vector with N values for N electrodes 
Elevation for each electrode is either relative or absolute
<code>   
topography = [z_1, z_2, .... z_Nel] 
</code>
where 1, 2, ... Nel are the electrodes indexes and z is their elevation.
If topography is negligible just use
<code>
topography = np.zeros(Nel)
</code>
Example of topography input (64 values for 64 electrodes):
'''
# topography=[70.2083781 , 70.25612936, 70.29714876, 70.33390108, 70.37253795,
#        70.42013276, 70.48375885, 70.56810439, 70.66831675, 70.77715812,
#        70.88739067, 70.99401378, 71.10097568, 71.21446183, 71.34065765,
#        71.48262625, 71.63094135, 71.77305433, 71.89641657, 71.99142024,
#        72.06022059, 72.1079137 , 72.1395956 , 72.16003001, 72.17265119,
#        72.18056108, 72.18686159, 72.19424756, 72.20378544, 72.2161346 ,
#        72.23195441, 72.2507575 , 72.26746962, 72.2758698 , 72.26973708,
#        72.24455895, 72.20265694, 72.14806104, 72.08480125, 72.01659512,
#        71.9459104 , 71.87490236, 71.80572631, 71.73903095, 71.66943862,
#        71.59006507, 71.49402606, 71.37944692, 71.2644913 , 71.17233244,
#        71.12614357, 71.13826896, 71.17773707, 71.2027474 , 71.17149943,
#        71.05677538, 70.88968824, 70.71593373, 70.58120754, 70.51859371,
#        70.51072949, 70.52764048, 70.53935227, 70.539]
topography=np.zeros(Nel)
pos = np.array([p[0] for p in data.sensorPositions()])
dx = np.min(np.diff(sorted(pos)))
elpos=[];Nel=data.sensorCount();
for i in range(Nel):
    print(i, data.sensorPosition(i))
    elpos.append(data.sensorPosition(i)[0])
elpos = make_topo(topography, elpos, Nel, dx)
fig,ax=plt.subplots(figsize=(8,1))
ax.plot(elpos[:,0],elpos[:,2]);ax.plot(elpos[:,0],elpos[:,2],'.');ax.set_aspect('equal')
data.setSensorPositions(elpos)

''' ---------------------------------
                END TOPOGRAPHY
------------------------------------'''

#%% Add error where zero error. 
err = np.array(data["err"])
# An estimate of average error is used in case some or all of the original data has no error
err[err <= 0] = 0.02 # here an estimate of 2% if a good ball park value
data["err"] = err

#%% save to new reformated file (optional):
data.save(infile[:-4]+"_unfied.dat")

#%% INVERSION START HERE:
# In the next cell, the formatted data is imported,  the manager is set up and the inversion is performed <br>
# A mesh is created on-the-fly based on the sensor positions <br>
# The `invert` function returns the model estimate. A number of options can be modified as needed:
# 
# ### OPTIONAL PARAMETERS AND SOME INDICATIVE VALUES:
# <code>
#  maxIter=12             | Will stop after maxIter cycles if not converged yet 
#  quality = 28           | Quality of the mesh. lower if code crashes.
#  useGradient=True       | Inital model has a vertical velocity gradient or not
#  vTop=150, vBottom=3000 | Initial model gradient is imposed by setting top and bottom velocities.
#                           Usually leave these out by commenting them with #
#  limits=[150,3000]      | Limits to max and min possible velocities in model.
#                           Use max min estimates from linear fit of arrival times.
#  paraMaxCellSize=10     | Limit the size of the cells in the mesh. Small look nicer but slower. 
#                           Increase if code crashes. Start testing with large values. 
#                           When happy with result, reduce to approximate geophone spacing. 
#  paraDepth=50           | Likely penetration depth of the survey, plus some
#  zWeight=0.1            | 0-1, smaller value will perfer layered solution. 
#  lam= 50                | Regularisation parameter, will prefer smoother (reduced curvature) solutions if large
#                           Large lam will also converge faster.
#  cType=1                | Set the type of mninimisation process, should not make a big difference
#  verbose=1              | Printout a lot of info if set to 1
# </code>

mgr = ert.ERTManager()
data['k'] = ert.createGeometricFactors(data, numerical=True)
model = mgr.invert(data, maxIter=12, quality = 28,
          useGradient=True, #vTop=150, vBottom=300, 
          #limits=[100,3800],
          paraMaxCellSize=0.25, paraDepth=16,
          zWeight=0.5, lam= 100, cType=2, 
          verbose=1)
#

#%% PLOTTING RESULTS:
mgr.showResult()#(coverage=ert.coverage())
fig2=plt.gcf()
ax2 = fig2.axes[0]
#ax2.plot(elpos[:,0],elpos[:,2],'ro',ms=3);
#fig2.axes[0].set(ylim=(-45.0, 0.0))
#fig2.set_size_inches(6,3);
#%%
fig2.savefig(infile[:-4]+'.pdf')
#%%
mgr.showData()
# -
