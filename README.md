To run online with jupyter lab:

https://mybinder.org/v2/gh/stefanazzz/Geophys3.4/main

with classic notebook:

https://mybinder.org/v2/gh/stefanazzz/Geophys3.4/main?filepath=Resistivity/2D/ERT2D_dat_res_3.4.ipynb

## Online testbench for geophysical imaging codes
### Includes *INTERACTIVE GRAPHICS* for picking and model design
Stefan Nielsen, Durham UNiversity, May 2024

### Uses
   - **pygimli** (https://www.pygimli.org/)
   - obspy (this is useful for i/o of seismic datafiles)
   - pandas (this is useful for i/o of spreadsheet data)
   - shapely (this is useful for interactive plotting/picking) 
   - etc. etc.
     
### Does
  - Interactive picking, AGC, reduced velocity on seismic shot gather
  - Reads dat and res file formats from TIGRE (Electrical Resistivity Tomography equipment)
  - Reads DAT and dat from smartseis and from geode seismic acquisision equipment
  - Inversion of res and seis data with pygimli modelling and inversion software
  - Interactive design of 2D models with polygons
  - Modelling of gravity data using pygimli
    
### aims:
- workable on jupyterlab / mybinder
- create sharable online version (and backup for Durham Geophysics class)
  
### steps for creation of local enviroment with conda:
-  conda create -n geo05 -c gimli -c conda-forge "pygimli>=1.5.0"
-  conda activate geo05
-  pip install jupyterlab
-  pip install pandas
-  pip install ipympl
-  conda install obspy
-  pip install shapely

### environment.yml file for MYBINDER:
<pre>
channels:
  - conda-forge
  - gimli
  - defaults
dependencies:
  - obspy
  - ipympl
  - pandas
  - shapely
  - pip:
    - pygimli
</pre>

### this also works, but not the preferred one as it requires downgrade to python 3.8: 
<pre>
channels:
  - conda-forge
  - gimli
  - defaults
dependencies:
  - python=3.8
  - pygimli>=1.1.0
  - obspy
  - ipympl
  - pandas
  - shapely
</pre>  
