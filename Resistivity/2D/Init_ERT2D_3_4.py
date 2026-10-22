import warnings
warnings.filterwarnings('ignore') # just to make it cleaner in the notebook
import numpy as np
#import pybert as pb
import pygimli as pg
from pygimli.physics import ert
import pygimli.meshtools as mt  # save space
import pandas as pd
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)
#############################
def read_dat(tigre_file):
    import numpy as np
    import pandas as pd
    # case of input .dat file:
    if tigre_file[-3:]=='dat':
        with open(tigre_file, "r") as filin:
            headr=filin.read().splitlines()[0:6]
        Nread=float(headr[3])
        dx=float(headr[1])
        filin.close()
        # read data until end of file and strip the ending zeroes:
        readings=[];
        with open(tigre_file, "r") as filin:
            dada=filin.read().splitlines()[6:-1]
        for line in dada:
            if line.strip("\n") != "0":
                readings.append(line.split())
        filin.close()
        readings=np.array(readings,dtype=np.float32)
        first_spacing=readings[0,1]
        Nel=2
        for ri in readings:
            Nel=Nel+1 # Count Number of electrodes
            if ri[1] != first_spacing:
                break
        #
        display(pd.DataFrame(headr))
        print("tot readings: ",Nread,"|| el. spacing dx= ",dx,"|| nr. of electrodes \033[1m Nel\033[0m = ",Nel)
        L=dx*(Nel-1)
        elpos = np.linspace(0.0,L,Nel)
    # case of input .res file:    
    elif tigre_file[-3:]=='res':
        he=pd.read_csv(tigre_file,sep=',',header=None,skiprows=5,nrows=1);he
        Nread=he.iloc[0][2]
        dx= he.iloc[0][5]
        Nel=he.iloc[0][7]
        df=pd.read_csv(tigre_file,sep=',',header=None,skiprows=7);df
        print("tot readings: ",Nread,"|| el. spacing dx= ",dx,"|| nr. of electrodes \033[1m Nel\033[0m = ",Nel)
        readings=np.array(df.iloc[:])  
        L=dx*(Nel-1)
        elpos = np.linspace(0.0,L,Nel)
    # error message if mistaken file name / format:
    else: 
        print("\033[1;31m NO GOOD! \033[0m file not found or wrong format? " )
        print('\033[1m I need a file name ending in .dat or a .res \033[0m')
        return([0,0,0,0,0])
    return(dx,Nread,Nel,elpos,readings)
#######################################
def make_topo(topo,elpos,Nel,dx):
    import numpy as np
    elpos = np.array([elpos,np.zeros(Nel),np.array(topo)]).T
    return(elpos)
def make_abmnr(filin,readings,dx,err_avg):
    if filin[-3:]=='dat':
        abmnr=[]
        for ree in readings:
            p1=int(round((ree[0]-ree[1]/2.)/dx))+1
            p2=int(round((ree[0]+ree[1]/2.)/dx))+1
            c1=int(round((ree[0]-3*ree[1]/2.)/dx))+1
            c2=int(round((ree[0]+3*ree[1]/2.)/dx))+1
            abmnr.append([c1,c2,p1,p2,ree[2],err_avg])
        return(abmnr)
    else:
        #p1_id=[];p2_id=[];c1_id=[];c2_id=[];rhoa=[];err=[];
        abmnr=[]
        for read in readings:
            xpos=float(read[3])
            x_a=float(read[1])*dx
            p1=int(round((xpos-x_a/2.)/dx))+1
            p2=int(round((xpos+x_a/2.)/dx))+1
            c1=int(round((xpos-3*x_a/2.)/dx))+1
            c2=int(round((xpos+3*x_a/2.)/dx))+1
            rhoa=float(read[14])
            err=abs(float(read[11]))
            if err == 0.:err=err_avg
            abmnr.append([c1,c2,p1,p2,rhoa,err])
        return(abmnr)
##############################
# def write_formatted_res():
#     filout=tigre_file[:-4]+"_uni2_res"+".csv" # name of output file
#     file_object = open(filout, 'w')
#     file_object.write(str(nElectrodes)+"\n") # write nr. of electrodes in array
#     file_object.write("# x y z \n")  # write comment line 
#     for val in elpos:  # write x, y, z pos for each electrode:
#         file_object.write(str(val[0])+"\t"+ str(val[1])+"\t"+ str(val[2])+"\n")
#     file_object.write(str(int(nReadings))+"\n") # write total nr. of readings
#     file_object.write('# a b m n err rhoa \n')  # comment line
#     for i in range(int(nReadings)):  # write index of current electrodes (a,b), potential electrodes (m,n), and app.resistivity (r) 
#         lin = str(c1_id[i])+"\t"+str(c2_id[i])+\
#          "\t"+str(p1_id[i])+"\t"+str(p2_id[i])+\
#          "\t"+str(err[i]) + "\t"+str(rhoa[i])
#         file_object.write(lin+"\n")
#     file_object.write(str(0)) # append zero to signal end of data file
    
#     # Close the file
#     file_object.close()
#     #????

def write_formatted(err,fil,Nel,elpos,Nread,abmnr):
    filout=fil[:-4]+"_uni2"+".csv" # name of output file
    file_object = open(filout, 'w')
    file_object.write(str(Nel)+"\n") # write nr. of electrodes in array
    file_object.write("# x y z \n")  # write comment line 
    for val in elpos:  # write x, y, z pos for each electrode:
        file_object.write(str(val[0])+"\t"+ str(val[1])+"\t"+ str(val[2])+"\n")
    file_object.write(str(int(Nread))+"\n") # write total nr. of readings
    file_object.write('# a b m n err rhoa \n')  # comment line
    for val in abmnr:  # write index of current electrodes (a,b), potential electrodes (m,n), and app.resistivity (r) 
        lin=str(val[0])+"\t"+str(val[1])+\
        "\t"+str(val[2])+"\t"+str(val[3])+\
        "\t"+str(val[5])+"\t"+str(val[4])
        file_object.write(lin+"\n")
    file_object.write(str(0)) # append zero to signal end of data file
    # Close the file
    file_object.close()
    return(filout)

def write_formatted_res(err,fil,Nel,elops,Nread,abmnr):
    filout=fil[:-4]+"_uni2"+".csv" # name of output file
    file_object = open(filout, 'w')
    file_object.write(str(Nel)+"\n") # write nr. of electrodes in array
    file_object.write("# x y z \n")  # write comment line 
    for val in elpos:  # write x, y, z pos for each electrode:
        file_object.write(str(val[0])+"\t"+ str(val[1])+"\t"+ str(val[2])+"\n")
    file_object.write(str(int(Nel))+"\n") # write total nr. of readings
    file_object.write('# a b m n err rhoa \n')  # comment line
    for i in range(int(Nread)):  # write index of current electrodes (a,b), potential electrodes (m,n), and app.resistivity (r) 
        lin = str(c1_id[i])+"\t"+str(c2_id[i])+\
         "\t"+str(p1_id[i])+"\t"+str(p2_id[i])+\
         "\t"+str(err[i]) + "\t"+str(rhoa[i])
        file_object.write(lin+"\n")
    file_object.write(str(0)) # append zero to signal end of data file
    
    # Close the file
    file_object.close()