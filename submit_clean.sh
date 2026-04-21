#!/bin/bash

# Clean the radical sandbox (pilot/task). 
rm -rf /home/jovyan/radical.pilot.sandbox/* \
       /home/jovyan/radical.pilot.sandbox/.[!.]* \
       /home/jovyan/radical.pilot.sandbox/..?*

# Clean Frontend radical session folder
rm -rf re.session.*