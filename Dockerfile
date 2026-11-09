# Use an official Python image as the base
FROM python:3.12-slim
WORKDIR /app

# Install apt dependencies
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install and configure Poetry
RUN curl -sSL https://install.python-poetry.org | python -
ENV PATH="/root/.local/bin:$PATH"
ENV POETRY_VIRTUALENVS_IN_PROJECT=true

# Install Python dependencies
COPY pyproject.toml poetry.lock /app/
# A GitHub token is required for private repositories. It is mounted as a build secret and the .netrc holding it is
# removed in the same step, so it is never stored in an image layer.
RUN --mount=type=secret,id=github_token,required=true \
    umask 077 && \
    trap 'rm -f /root/.netrc' EXIT && \
    printf 'machine github.com login git password %s\n' "$(cat /run/secrets/github_token)" > /root/.netrc && \
    poetry install --no-root --only main,deployment
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code and install app
COPY . /app/
RUN poetry install --only-root

ENTRYPOINT ["poetry", "run", "dtc-wrapper"]
