FROM quay.io/hdc-workflows/ubuntu:20.04

# bust cache
# ADD http://date.jsontest.com /etc/builddate

LABEL maintainer "Jared Galloway <jgallowa@fredhutch.rg>" \
      version "0.1.9" \
      description "multidms"

# install needed tools
RUN apt-get update --fix-missing -qq && \
    DEBIAN_FRONTEND=noninteractive \
    apt-get install -y -q \
    locales \
    libncurses5-dev  \
    libncursesw5-dev \
    build-essential \
    pkg-config \
    zlib1g-dev \
    python3 \
    python3-pip \ 
    python3-venv

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# install multidms
RUN pip install multidms==0.1.9
