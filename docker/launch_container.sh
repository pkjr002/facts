#!/bin/bash


#--------------------------------------------------------------------
# Check Docker daemon status

set -e
if docker info >/dev/null 2>&1; then
  echo "Docker daemon is running."
else
  echo "Docker daemon is not running... launch it." >&2
  exit 1
fi


#--------------------------------------------------------------------
# Build docker Image 

TAG="alt-emis"

echo "Building docker Image $TAG"
docker build --no-cache --target facts-core -t "$TAG" .

# TAG1="facts-jupyter"
# echo "Build docker Image $TAG1"
# docker build --no-cache --target facts-jupyter -t "$TAG1" .


#--------------------------------------------------------------------
# Create a docker volume for radical sandbox
sandbox="facts_sandbox"
docker volume create "$sandbox"


#--------------------------------------------------------------------
# Launch a docker Container

facts_folder = "/link/to/facts"
facts_modules-data = "/link/to/facts-/modules-data"

# # ==> Linux/Mac: 
docker run -it --init --shm-size=2g \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -e HDF5_USE_FILE_LOCKING=FALSE \
    --volume="$facts_folder":/opt/facts:delegated \
    --volume=facts_sandbox:/home/jovyan/radical.pilot.sandbox \
    --volume="$facts_modules":/opt/facts/modules-data \
    -w /opt/facts \
    "$TAG" /bin/bash