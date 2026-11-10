FROM rocker/geospatial AS base
WORKDIR /app

# Install R packages
# Need to increase timeout because INLA is a large download
ENV R_DEFAULT_INTERNET_TIMEOUT=600
RUN Rscript -e 'install.packages(\
    c("ggplot2", "sf", "withr", "INLA"), \
    repos=c(getOption("repos"), INLA="https://inla.r-inla-download.org/R/stable"), \
    dep=TRUE)'

# Verify R package installation
RUN Rscript -e 'library(ggplot2); library(sf); library(withr); library(fmesher); library(INLA)'

# Install Python 3.13
RUN apt-get update && apt-get -y install --no-install-recommends software-properties-common openssh-client
RUN add-apt-repository -y ppa:deadsnakes/ppa
RUN apt-get update && apt-get -y install --no-install-recommends \
    python3.13 \
    python3.13-dev \
    curl \
    build-essential
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.13 /usr/bin/python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install and configure Poetry
RUN curl -sSL https://install.python-poetry.org | python3.13 -
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Install Python dependencies
COPY pyproject.toml poetry.lock /app/
ENV LD_LIBRARY_PATH="/usr/local/lib/R/lib:/usr/local/lib:/usr/lib/x86_64-linux-gnu"
ENV PATH="/app/.venv/bin:$PATH"

FROM base AS test
RUN poetry install --no-root --with dev --without deployment

FROM test AS devcontainer
COPY . /app/

FROM base AS deployment
RUN --mount=type=secret,id=github_token,required=true \
    umask 077 && \
    trap 'rm -f /root/.netrc' EXIT && \
    printf 'machine github.com login git password %s\n' "$(cat /run/secrets/github_token)" > /root/.netrc && \
    poetry install --no-root --with deployment --without dev

# Copy application code and install app
COPY . /app/
RUN poetry install --only-root

CMD ["poetry", "run", "dtc-wrapper"]
