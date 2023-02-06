#!/bin/bash

FILENAME='dummies_1.5.6.tar.gz'
if [ -f "$FILENAME" ]; then
  echo "$FILENAME present, skip Download"
else
  echo "$FILENAME Downloading ..."
  wget "https://cran.r-project.org/src/contrib/Archive/dummies/$FILENAME"
fi

Rscript emulandice_bundle_dependencies.R
tar cvzf emulandice_bundled_dependencies.tgz .Rprofile packrat/
