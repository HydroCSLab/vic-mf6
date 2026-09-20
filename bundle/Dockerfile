# syntax=docker/dockerfile:1.7

ARG UBUNTU_VERSION=24.04

FROM ubuntu:${UBUNTU_VERSION} AS builder

ARG DEBIAN_FRONTEND=noninteractive

RUN printf 'Acquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99force-ipv4 \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        gfortran \
        libnetcdf-dev \
        libnetcdff-dev \
        libopenmpi-dev \
        libpetsc-real-dev \
        meson \
        ninja-build \
        pkg-config \
        python3 \
        python3-dev \
        python3-pip \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY bundle/components/vic /src/vic
COPY bundle/components/modflow6 /src/modflow6
COPY pyproject.toml README.md COPYING MANIFEST.in /src/vic-mf6/
COPY src /src/vic-mf6/src
COPY native /src/vic-mf6/native
COPY scripts /src/vic-mf6/scripts
COPY examples /src/vic-mf6/examples
COPY vicmf6 /src/vic-mf6/vicmf6
COPY bundle/components.lock /tmp/components.lock
COPY bundle/requirements-container.txt /tmp/requirements-container.txt
COPY bundle/requirements-container.lock /tmp/requirements-container.lock

# vic links directly against the same openmpi and netcdf libraries used by
# the rest of the stack. a stable version string keeps the binary provenance
# independent of transient docker build metadata.
RUN vic_revision=$(awk -F '|' '$1 == "vic" {print $2}' /tmp/components.lock) \
    && test -n "$vic_revision" \
    && make -C /src/vic/vic/drivers/image model \
        GIT_VERSION="$vic_revision" \
        HOSTNAME=vic-mf6-container \
        LOG_LVL=30 \
        USER=vicmf6 \
    && install -D -m 0755 \
        /src/vic/vic/drivers/image/vic_image.exe \
        /opt/vicmf6/bin/vic_image.exe

# the parallel build produces both the executable and the xmi shared library.
# the coupler loads the library while the executable remains useful for direct
# diagnostics and for independent modflow 6 runs inside the image.
# use the mpi wrapper as the Fortran linker. MODFLOW's parallel sources call
# both mpi_f08 and legacy mpi entry points; the wrapper supplies the complete
# OpenMPI Fortran link set in the order required by the distribution.
RUN FC=mpifort meson setup /tmp/modflow6-build /src/modflow6 \
        --buildtype=release \
        --prefix=/opt/vicmf6 \
        --bindir=bin \
        --libdir=lib \
        -Dparallel=true \
    && meson compile -C /tmp/modflow6-build \
    && meson install -C /tmp/modflow6-build

# build the small preload library once, with the image's openmpi compiler.
# runtime acceptance then exercises installed software without compiling code.
RUN /src/vic-mf6/native/build_disconnect.sh \
    && install -D -m 0755 \
        /src/vic-mf6/native/libvic_parent_disconnect.so \
        /opt/vicmf6/lib/libvic_parent_disconnect.so

RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python -m pip install --no-cache-dir \
        pip==25.2 \
        setuptools==80.9.0 \
        wheel==0.45.1 \
    && MPICC=mpicc /opt/venv/bin/python -m pip install --no-cache-dir \
        --no-binary=mpi4py \
        --constraint /tmp/requirements-container.lock \
        --requirement /tmp/requirements-container.txt \
    && /opt/venv/bin/python -m pip install --no-cache-dir \
        --no-build-isolation \
        --no-deps \
        /src/vic-mf6


FROM ubuntu:${UBUNTU_VERSION} AS runtime

ARG DEBIAN_FRONTEND=noninteractive
ARG USER_ID=1000
ARG GROUP_ID=1000
ARG SOURCE_REPOSITORY=https://github.com/mabdazzam/vic-mf6
ARG VERSION=local
ARG REVISION=unknown

RUN printf 'Acquire::ForceIPv4 "true";\n' > /etc/apt/apt.conf.d/99force-ipv4 \
    && apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        libnetcdf-dev \
        libnetcdff-dev \
        libopenmpi-dev \
        libpetsc-real-dev \
        openmpi-bin \
        python3 \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/vicmf6 /opt/vicmf6

# retain the exact source used for every installed component. this keeps the
# image inspectable and preserves the license files beside each project.
COPY bundle/components/vic /opt/vicmf6/src/vic
COPY bundle/components/modflow6 /opt/vicmf6/src/modflow6
COPY pyproject.toml README.md COPYING MANIFEST.in /opt/vicmf6/src/vic-mf6/
COPY src /opt/vicmf6/src/vic-mf6/src
COPY native /opt/vicmf6/src/vic-mf6/native
COPY scripts /opt/vicmf6/src/vic-mf6/scripts
COPY examples /opt/vicmf6/src/vic-mf6/examples
COPY tests /opt/vicmf6/src/vic-mf6/tests
COPY vicmf6 /opt/vicmf6/src/vic-mf6/vicmf6
COPY bundle/components.lock /opt/vicmf6/share/component-revisions.txt
COPY bundle/requirements-container.txt /opt/vicmf6/share/requirements-container.txt
COPY bundle/requirements-container.lock /opt/vicmf6/share/requirements-container.lock
COPY bundle/examples/stehekin /opt/vicmf6/examples/stehekin
COPY bundle/scripts/container-entrypoint.sh /opt/vicmf6/bin/container-entrypoint

RUN if getent group "${GROUP_ID}" >/dev/null; then \
        runtime_group=$(getent group "${GROUP_ID}" | cut -d: -f1); \
    else \
        groupadd --gid "${GROUP_ID}" vicmf6; \
        runtime_group=vicmf6; \
    fi \
    && useradd --create-home --non-unique --uid "${USER_ID}" --gid "$runtime_group" vicmf6 \
    && chmod 0755 /opt/vicmf6/bin/container-entrypoint \
    && mkdir -p /results /work /tmp/matplotlib \
    && chown -R vicmf6:"$runtime_group" /results /work /tmp/matplotlib

ENV PATH="/opt/venv/bin:/opt/vicmf6/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/vicmf6/lib" \
    MPLCONFIGDIR="/tmp/matplotlib" \
    PYTHONDONTWRITEBYTECODE="1" \
    PYTHONUNBUFFERED="1" \
    PYTHONWARNINGS="ignore::DeprecationWarning"

LABEL org.opencontainers.image.title="VIC-MF6 two-way coupling stack" \
      org.opencontainers.image.description="Reproducible VIC and MODFLOW 6 two-way coupling runtime" \
      org.opencontainers.image.licenses="GPL-3.0-or-later" \
      org.opencontainers.image.source="${SOURCE_REPOSITORY}" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.revision="${REVISION}"

USER vicmf6
WORKDIR /work
ENTRYPOINT ["/opt/vicmf6/bin/container-entrypoint"]
CMD ["acceptance"]
