FROM ubuntu:focal AS facts-core

# set env variables for RCT and apt
ENV DEBIAN_FRONTEND="noninteractive"

# set shell for RUN instruction
SHELL [ "/bin/bash", "-c" ]

# installs dependences
RUN apt-get update &&\
    apt-get install -y \
        python3.9 \
        python3-pip \
        python3.8-venv \
        git \
        sudo \
        libhdf5-dev \
        libnetcdf-dev \
        gnupg \
        curl \
        apt-transport-https \
        dnsutils \
        iputils-ping

# Creates and activates python3 virtual environment
RUN python3 -m venv --system-site-packages factsVe &&\
    source factsVe/bin/activate

# installs required FACTS packages
RUN pip install --no-cache-dir --upgrade \
        setuptools==69.0.2 pip==23.3.1 wheel==0.42.0

RUN pip install --no-cache-dir \
        radical.entk==1.52.0 \
        radical.gtod==1.52.0 \
        radical.pilot==1.52.1 \
        radical.saga==1.52.0 \
        radical.utils==1.52.0 \
        pyyaml==6.0.1 \
        xarray==2023.1.0 \
        numpy==1.24.4 \
        netcdf4==1.6.5 \
        h5py \
        dask

        
# installs R for ubuntu 20.04 Focal
RUN apt-get update &&\
    apt-get install -y --no-install-recommends r-base \
        cmake \
        libopenblas-dev \
        gfortran


ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}

# Make sure the contents of our repo are in ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

# make directory for radical pilot sandbox
RUN mkdir -p ~/radical.pilot.sandbox

# Activate venv
RUN echo -e ". /factsVe/bin/activate" > ~/.bashrc

# create resource_local.json with the specified contents as well as
# create resource_local.json with the specified contents
RUN mkdir -p ~/.radical/pilot/configs
RUN printf '{\n\
    "local": {\n\
        "rp_version"    : "installed",\n\
        "virtenv_mode"  : "use"\n\
    }\n\
}' > ~/.radical/pilot/configs/resource_local.json

# return a shell to the user
CMD /bin/bash

FROM facts-core AS facts-jupyter

USER root
RUN pip install --no-cache-dir \
	matplotlib==3.7.4 \
	notebook==7.0.6 \
	jupyterlab==4.0.9 \
	jupyter==1.0.0

EXPOSE 8888
USER ${NB_USER}

#ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root"]
