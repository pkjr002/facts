#!/bin/bash

# Clean the radical sandbox (pilot/task). 
if rm -rf /home/$(whoami)/radical.pilot.sandbox/* \
          /home/$(whoami)/radical.pilot.sandbox/.[!.]* \
          /home/$(whoami)/radical.pilot.sandbox/..?*; then
    echo "Successfully cleaned the radical sandbox."
else
    echo "Notice: Sandbox was already empty or encountered an issue."
fi

# Clean Frontend radical session folder
if rm -rf re.session.*; then
    echo "Successfully cleaned frontend radical session folders."
fi