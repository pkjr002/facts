#!/usr/bin/env bash
set -euo pipefail


#-STEP 0-------------------------------------------------------------
#         Config.
#--------------------------------------------------------------------
IMAGE="${IMAGE:-facts_io}"
IMAGE1="${IMAGE1:-}"   # optional; set to build jupyter target

facts_folder="${facts_folder:-/Users/pk695/Desktop/facts_test/facts_IO}"
facts_modules_data="${facts_modules_data:-/Users/pk695/werk.M2/FACTS_dev/2401_RFF.SPs/facts_development/facts/modules-data}"

MODE="${MODE:-full}"   # full | run
sandbox="${sandbox:-tmp}"  # tmp | docker_volume_sandbox


# ----------------------------
# Helpers
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
    *) die "Unknown sandbox='$sandbox'";;
  esac

  # Common args (arrays avoid quoting bugs)
  local -a run_args=(
    -it --init --shm-size=2g
    -e OMP_NUM_THREADS=1
    -e OPENBLAS_NUM_THREADS=1
    -e MKL_NUM_THREADS=1
    -e NUMEXPR_NUM_THREADS=1
    -e HDF5_USE_FILE_LOCKING=FALSE
    --volume "${facts_folder}/facts:/opt/facts:delegated"
    --volume "${facts_modules_data}:/opt/facts/modules-data"
    -w /opt/facts
  )

  docker run "${run_args[@]}" $sandbox_mount "$IMAGE" /bin/bash
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