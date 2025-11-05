#!/bin/bash
set -euo pipefail


#-STEP 0-------------------------------------------------------------
#         Config.
#--------------------------------------------------------------------

IMAGE="alt-emis-nu"
# IMAGE1="facts-jupyter"  # uncomment if you want to build jupyter image. 

facts_folder="path/to/facts"
facts_modules_data="path/to/facts/modules-data"

MODE="${MODE:-full}"   # default="full", use "run" to spin up more containers


#-STEP 1-------------------------------------------------------------
#         Check Docker daemon status.
#--------------------------------------------------------------------
printf '\n\n'

if docker info >/dev/null 2>&1; then
  echo "Docker daemon is running."
else
  echo "Docker daemon is not running... launch it." >&2
  exit 1
fi
printf '\n\n'



#-STEP 2-------------------------------------------------------------
#         Build docker Image. 
#--------------------------------------------------------------------
if [ "$MODE" = "full" ]; then
  echo "Building docker Image $IMAGE"
  docker build --no-cache --target facts-core -t "$IMAGE" .

  # Optional:: If IMAGE1 is present, build the jupyter-facts
  if [ -n "${IMAGE1:-}" ] && [ -n "$(printf '%s' "$IMAGE1" | tr -d '[:space:]')" ]; then
    echo "Build docker Image $IMAGE1"
    docker build --no-cache --target facts-jupyter -t "$IMAGE1" .
  fi
  printf '\n\n'
fi


#-STEP 3-------------------------------------------------------------
#         Create a docker volume for radical sandbox.
#--------------------------------------------------------------------
sandbox="facts_sandbox"

if docker volume inspect "$sandbox" >/dev/null 2>&1; then
  echo "Volume '$sandbox' already exists, skipping creation."
else
  if docker volume create "$sandbox" >/dev/null 2>&1; then
    echo "Created volume '$sandbox'."
  else
    echo "ERROR: Failed to create volume '$sandbox'." >&2
    exit 1
  fi
fi
printf '\n\n'



#-STEP 4-------------------------------------------------------------
#         Launch a docker Container.
#--------------------------------------------------------------------

cat <<'EOF'
#############################################
#                                           #
#   Welcome to the FACTS docker container   #
#                                           #
#############################################
EOF
printf '\n\n'

# # ==> Linux/Mac: 
docker run -it --init --shm-size=2g \
    -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e NUMEXPR_NUM_THREADS=1 \
    -e HDF5_USE_FILE_LOCKING=FALSE \
    --volume="$facts_folder":/opt/facts:delegated \
    --volume=facts_sandbox:/home/jovyan/radical.pilot.sandbox \
    --volume="$facts_modules_data":/opt/facts/modules-data \
    -w /opt/facts \
    "$IMAGE" /bin/bash