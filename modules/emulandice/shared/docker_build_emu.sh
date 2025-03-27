#!/bin/bash

rm -v -f emulandice_*gz* emulandice_bundled_dependencies.tgz .Rprofile
rm -v -fr packrat/
Rscript emulandice_bundle_dependencies.R
tar cvzf emulandice_bundled_dependencies.tgz .Rprofile packrat/