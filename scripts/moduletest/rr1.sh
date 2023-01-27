#!/bin/bash

# Choose the test case. (UN comment one)
#mod=fair ;                submod=temperature ;
#mod=tlm ;                 submod=oceandynamics ; 
#mod=ipccar5;              submod=thermalexpansion;
#mod=kopp14 ;              submod=glaciers ;
#mod=ipccar5;              submod=glaciers;
#mod=ipccar5;              submod=glacierscmip6;
#mod=ipccar5;              submod=glaciersfair;
#mod=ipccar6;              submod=gmipemuglaciers;
mod=emulandice;           submod=glaciers;    # <==
#mod=ipccar5;              submod=icesheets;
#mod=bamber19;             submod=icesheets;
#mod=ipccar6;              submod=ismipemuicesheets;
#mod=kopp14;               submod=icesheets;
#mod=FittedISMIP;          submod=GrIS;
#mod=emulandice;           submod=GrIS;  # <==
#mod=deconto16;            submod=AIS;    
#mod=deconto21;            submod=AIS;
#mod=ipccar6;              submod=larmipAIS;
#mod=larmip;               submod=AIS;
#mod=kopp14;               submod=landwaterstorage;
#mod=ssp;               submod=landwaterstorage;

#mod=FittedISMIP ;         submod=icesheet ; 
#mod=NZInsarGPS ;          submod=verticallandmotion ;
#mod=ar5 ;                 submod=thermalexpansion  ;    #glaciers, glacierscmip6, glaciersfair, icesheets, thermalexpansion 

#mod=dp16 ;                submod=icesheet ; 
#mod=dp21 ;                submod=icesheet ; 

#mod=emulandice ;          submod=AIS ;  #heavy errors
#mod=emulandice ;          submod=glaciers ;
#mod=emulandice ;          submod=GrIS ;

#mod=extremesealevel ;     submod=pointsoverthreshold ; #need an ip file

#mod=genmod ;              submod=directsample ;

#mod=ipccar6 ;             submod=bambericesheet ;
#mod=ipccar6 ;             submod=gmipemuglaciers ;
#mod=ipccar6 ;             submod=ismipemuicesheet ; 
#mod=ipccar6 ;             submod=larmipicesheet ;


#mod=kopp14 ;              submod=icesheets ;
#mod=kopp14 ;              submod=landwaterstorage ;
#mod=kopp14 ;              submod=oceandynamics ;
#mod=kopp14 ;              submod=thermalexpansion ;
#mod=kopp14 ;              submod=verticallandmotion ;

#mod=larmip ;              submod=icesheet ;
#mod=ssp ;                 submod=landwaterstorage ;

#mod=total


# Run the specific testcase.

# If using SLURM
rm slurm-*
sbatch --export=MOD=$mod,SUBMOD=$submod run_moduletestG.sh





: << 'Issues'
==> Can we call the location file from a global main dir.



Issues