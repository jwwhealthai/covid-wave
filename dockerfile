# set base image
FROM ubuntu:20.04

# set the working directory in the container
WORKDIR /code

# system packages
RUN apt-get update && apt-get install -y curl

# Install miniconda to /miniconda
RUN curl -LO http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b
RUN rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:${PATH}
RUN conda update -y conda

# Install pip before cloning pymc3 repo and installing requirements
RUN conda install -c conda-forge -y pip
RUN git clone https://github.com/pymc-devs/pymc3 && cd pymc3 ; \
    pip install -r requirements.txt

# Clone and install sunode
RUN git clone https://github.com/asseyboldt/sunode && cd sunode ; \
    conda install --only-deps sunode ; pip install -e .


# #copy the content of the local src directory to the working directory
# COPY (want to copy the models folder incl subdirs across)

# # command to run on container start
# CMD