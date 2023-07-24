#!/bin/bash

# MAtcg the old version 
cp NZ_with_islands_VLM-v2.txt NZ_2km.txt

# Create the tar ball for amarel
tar -czvf NZInsarGPS1_verticallandmotion_preprocess_data.tgz NZ_2km.txt 