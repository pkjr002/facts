#!/usr/bin/env bash

# -Usage----------------------------------------------------------------
# This script creates and/or launches a FACTS Docker image/container.
#
# Run:
#   source launch_container.sh
#
# Before running, 
# review and update the user configuration in STEP 0,
# ==> especially:
#   IMAGE , container_name, CPU, memory, facts_folder, 
#   facts_modules_data, sandbox_path
# -------------------------------------------------------------------

set -euo pipefail


#-STEP 0-------------------------------------------------------------
#         User configuration.
#--------------------------------------------------------------------

# Main Docker image name
IMAGE="${IMAGE:-src}"
IMAGE1="${IMAGE1:-}"   # optional; set to build jupyter target

# Container name & details
container_name="${container_name:-src0406}"
CPU="${CPU:-12}"              # CPU (in terminal , linux: nproc    , mac:`sysctl hw.ncpu`)
memory="${memory:-30g}"       # RAM (in terminal , linux: free -h  , mac:`system_profiler SPHardwareDataType | grep "Memory:"`)  
shm_size="${shm_size:-2g}"

# Path to FACTS working directory
facts_folder="${facts_folder:-/scratch/usr/facts_Dev/202603_SRC}"

# Path to FACTS modules-data directory
facts_modules_data="${facts_modules_data:-/scratch4/modules-data}"

# Select one mode only:
#   full = build the image, then launch the container
#   run  = launch the container using an existing image
MODE="${MODE:-full}"     
# MODE="${MODE:-run}"    

# Sandbox options
sandbox="${sandbox:-sandbox_path}"   
sandbox_path="${sandbox_path:-/scratch4/radical.pilot.sandbox}"
# sandbox="${sandbox:-tmp}"  # tmp | docker_volume_sandbox

#- End of user configuration-----------------------------------------
#  X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X X
# -------------------------------------------------------------------


#-Helpers---------------------
# 
# ----------------------------
die() { echo "ERROR: $*" >&2; exit 1; }

banner() {
cat <<'EOF'
#############################################
#                                           #
#   Welcome to the FACTS docker container   #
#                                           #
#############################################
EOF
}


#-STEP 1-------------------------------------------------------------
#         Check Docker daemon status.
#--------------------------------------------------------------------
require_docker() {
  docker info >/dev/null 2>&1 || die "Docker daemon is not running... launch it."
  echo "Docker daemon is running."
}


#-STEP 2-------------------------------------------------------------
#         Build docker Image. 
#--------------------------------------------------------------------
build_images() {
  echo "Building docker Image $IMAGE"
  docker build --no-cache --target facts-core -t "$IMAGE" .

  if [[ -n "$IMAGE1" ]]; then
    echo "Building docker Image $IMAGE1"
    docker build --no-cache --target facts-jupyter -t "$IMAGE1" .
  fi
}


#-STEP 3-------------------------------------------------------------
#         Create a docker volume for radical sandbox.
#--------------------------------------------------------------------
ensure_sandbox_volume() {
  [[ "$sandbox" == "docker_volume_sandbox" ]] || return 0

  if docker volume inspect facts_sandbox >/dev/null 2>&1; then
    echo "Volume 'facts_sandbox' already exists, skipping creation."
  else
    docker volume create facts_sandbox >/dev/null || die "Failed to create volume 'facts_sandbox'."
    echo "Created volume 'facts_sandbox'."
  fi
}



#-STEP 4-------------------------------------------------------------
#         Launch a docker Container.
#--------------------------------------------------------------------
run_container() {
  local sandbox_mount
  case "$sandbox" in
    docker_volume_sandbox)
      sandbox_mount="--volume=facts_sandbox:/home/jovyan/radical.pilot.sandbox"
      ;;
    tmp)
      sandbox_mount="--volume=${facts_folder}/tmp/radical.pilot.sandbox:/home/jovyan/radical.pilot.sandbox"
      ;;
    sandbox_path)
      sandbox_mount="--volume=${sandbox_path}:/home/jovyan/radical.pilot.sandbox"
      ;;
    *) die "Unknown sandbox='$sandbox'";;
  esac

  # Common args (arrays avoid quoting bugs)
  local -a run_args=(
    -it --init 
    --name "$container_name"
    --cpus "$CPU"
    --memory "$memory"
    --memory-swap "$memory"
    --shm-size "$shm_size"
    -e HDF5_USE_FILE_LOCKING=FALSE
    --volume "${facts_folder}/facts:/opt/facts"
    --volume "${facts_modules_data}:/opt/facts/modules-data:ro"
    -w /opt/facts
  )

  docker run "${run_args[@]}" "$sandbox_mount" "$IMAGE" /bin/bash
}


#-STEP 5-------------------------------------------------------------
#         MAIN.
#--------------------------------------------------------------------
printf '\n\n'
require_docker
printf '\n\n'

case "$MODE" in
  full)
    build_images
    printf '\n\n'
    ;;
  run) ;;
  *)
    die "Unknown MODE='$MODE' (expected 'full' or 'run')"
    ;;
esac

ensure_sandbox_volume
printf '\n\n'

banner
printf '\n\n'
run_container