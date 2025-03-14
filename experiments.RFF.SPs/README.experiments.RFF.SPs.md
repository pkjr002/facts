# This is a folder containing expariments using RFF 3 gas. 


1. SandBox Folder
    - `/Users/pk695/werk.M2/FACTS_dev/2401_RFF.SPs/facts_development/tmp/radical.pilot.sandbox/`    


1. `debug_rffLL_project` local sandbox from the `pilot` to debug project script   

1. `glo.ssp585.rff` used to run the EPA case

1. `global.ssp245` fair ssp 245 run    

1. `global.ssp585`    

1. `rff.global.ssp245`     


<br>  

# Error Log   
**Experiment:** Alternate emissions  (base-emis ssp245)   
**nsamps:** 9999   
**Deets:** pyear_start: 2020; pyear_end: 2150.

1.  `rff.LL`  Currently working with this
    - **local only** 
    - <mark>ERROR</mark> hanging and not cteating `rff.LL.total.facts.total.wf2f.local`    
      the file is created but not in the totalled section and doesnot transfer.       
    - Sandbox located at `re.session.3440c078-aba6-11ef-9280-0242ac110002`  
    <br>

1.  `rff.LL.glo` replicate the above to diagnose error.    
    - **global only**    
    - Sandbox located at `re.session.713b7490-abf2-11ef-b155-0242ac110002`    
    - <span style="background-color: green;"> pass </span> 
    - Log
        - 2024-11-26 07:32:04  to    
        - 2024-11-26 08:36:51 Update: rff.LL.glo.total.facts.total.wf3e.global state: DONE
        - 2024-11-26 08:37:17 All components terminated
        <br>

1.  `rff.LL.lo` replicate the above to diagnose error.    
    - **local only** <mark>without wf2f</mark>    
    - Sandbox located at `re.session.179ae258-abfd-11ef-b9d4-0242ac110002`          
    - <span style="background-color: green;"> pass </span>    
    - Log (fill log lated at `./README.terminal.log.re.session.179ae258-abfd-11ef-b9d4-0242ac110002`)    
        - 2024-11-26 08:48:42 Update: rff.LL.lo.temperature.fair.rffLL state: SCHEDULING
        - 2024-11-27 04:49:47 Update: rff.LL.lo.total.facts.total.wf2e.global.workflow.task1 state: SUBMITTING     
            2024-11-27 14:49:51 All components terminated     
    <br>

    








