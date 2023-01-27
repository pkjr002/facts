#!/bin/bash

# Choose the test case. (UN comment one)
#mod=FittedISMIP ;         submod=icesheet ; 
#mod=NZInsarGPS ;          submod=verticallandmotion ;
#mod=ar5 ;                 submod=thermalexpansion  ;    #glaciers, glacierscmip6, glaciersfair, icesheets, thermalexpansion 

#mod=dp16 ;                submod=icesheet ; 
#mod=dp21 ;                submod=icesheet ; 

#mod=emulandice ;          submod=AIS ;  #heavy errors
#mod=emulandice ;          submod=glaciers ;
#mod=emulandice ;          submod=GrIS ;

#mod=extremesealevel ;     submod=pointsoverthreshold ; #need an ip file
#mod=fair ;                submod=temperature ;
#mod=genmod ;              submod=directsample ;

#mod=ipccar6 ;             submod=bambericesheet ;
#mod=ipccar6 ;             submod=gmipemuglaciers ;
#mod=ipccar6 ;             submod=ismipemuicesheet ; 
#mod=ipccar6 ;             submod=larmipicesheet ;

#mod=kopp14 ;              submod=glaciers ;
#mod=kopp14 ;              submod=icesheets ;
#mod=kopp14 ;              submod=landwaterstorage ;
#mod=kopp14 ;              submod=oceandynamics ;
#mod=kopp14 ;              submod=thermalexpansion ;
#mod=kopp14 ;              submod=verticallandmotion ;

#mod=larmip ;              submod=icesheet ;
#mod=ssp ;                 submod=landwaterstorage ;
#mod=tlm ;                 submod=oceandynamics ;
#mod=total


# Run the specific testcase.

# If using SLURM
rm slurm-*
sbatch --export=MOD=$mod/$submod run_moduletestG.sh





: << 'Issues'
==> Can we call the location file from a global main dir.



Issues