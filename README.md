![example workflow](https://github.com/dtc-ice-sheets/use_case_1/actions/workflows/use_case_1_test.yml/badge.svg)
![example workflow](https://github.com/dtc-ice-sheets/use_case_1/actions/workflows/use_case_1_deploy.yml/badge.svg)

# use_case_1

DTC-IS Hydropower modules aimed to predict climatically driven changes to freshwater flux from the Greenland Ice Sheet into the proglacial environment.

**Please familiarise yourself with the [DTC Ice Sheets Development Guide](https://dtc-ice-sheets.github.io/dtc_python_template/) before proceeding.**


## Installation

Due to the dependency on R packages, by far the easiest way to use this module is inside of a Docker container, using the provided Dockerfile.

Start by [installing and setting up Docker](https://docs.docker.com/engine/install/).

Then, clone the repository:
```bash
git clone https://github.com/dtc-ice-sheets/use_case_1.git
cd use_case_1
```

You will need a [GitHub personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#fine-grained-personal-access-tokens) with access to the DTC Ice Sheets private repositories for the Docker build process to install some of the Python dependencies:
```bash
echo "YOUR GITHUB TOKEN HERE" > .gh_token
```

The token is mounted only during dependency installation and is not stored in the image.
The Docker build context excludes local credentials and virtual environments.

The shared Dockerfile provides three build targets:

- `devcontainer`: development dependencies and source code, selected automatically by VSCode.
- `test`: development dependencies without source code, used by CI.
- `deployment`: deployment dependencies and the installed application, the default target.

If using VSCode, the rest of the setup is made very easy by the "dev containers" feature.

### Option 1: Setup using VSCode dev containers (highly recommended!)

1. Open the cloned repository folder in VSCode.
2. Click the blue box at the very bottom left of the VSCode window, and then click "Dev Containers: Reopen in Container". \
 For more guidance on this, see the [dev containers documentation](https://code.visualstudio.com/docs/devcontainers/containers).
3. Wait for the Docker setup to finish (might take a few minutes).
4. There should be a popup in the bottom right asking you to install recommended extensions, click "Yes". Otherwise go to the "Extensions" tab and install all the extensions that are marked as recommended.
5. Open a Python file, and ensure in the very bottom right of the VSCode window that the Python environment is set to "use-case-1-py3.13", located at `/app/.venv/bin/python`.
6. Click on the "Testing" tab icon on the left of the VSCode window. From there, click "Refresh Tests", and then click "Run Tests". After a short while you should see that all test have passed with a green check.

And that's it!

Note that the list of VSCode projects should now show two copies of this project: one labelled as "dev container", and one not.
You should ignore the normal project and make sure to always do all development work inside the dev container project.
If you are in the correct project the blue box in the bottom left should start with "Dev Container: ...".
If it doesn't, repeat Step 2 to switch to the correct project.

If there were no issues, you can now skip to the "Development" section.

### Option 2: Manual Docker setup

This section documents how to setup a Docker environment manually, in case VSCode with dev containers is not available for any reason.

The first step is to build the container, providing the GitHub token as a secret:
```bash
docker build --target deployment --secret id=github_token,src=.gh_token -t dtc-ice-sheets-use-case-1 .
```

For a development or CI image, replace `--target deployment` with `--target devcontainer`
or `--target test`, respectively. Development and test images exclude the deployment
dependency group; deployment images exclude the development dependency group.

Once this completes, the package module CLI can be used as follows:
```bash
docker run dtc-ice-sheets-use-case-1 python -m use_case_1.grf_module_entrypoint --mesh_name <mesh_name> --data_path <data_path> --output_path <output_path>
```

For more help, run:
```bash
docker run dtc-ice-sheets-use-case-1 python -m use_case_1.grf_module_entrypoint -h
```

You can also start an interactive shell session within the docker container:
```bash
docker run -it dtc-ice-sheets-use-case-1 bash
```

## Development

Install the dev dependencies:
```bash
poetry install --with dev
```

Update LD_LIBRARY_PATH so that R has access to its internal libraries.
```bash
export LD_LIBRARY_PATH="$(python -m rpy2.situation LD_LIBRARY_PATH):${LD_LIBRARY_PATH}"
```

Next, install the pre-commit hooks:
```bash
pre-commit install
```

Now, you should be able to run and pass the precommit hooks in your environment:
```bash
pre-commit
```

Note: If in need to push code that's not passing pre-commit's checks you can still commit using the --no-verify flag

```bash
git commit --no-verify -m "Commit message"
```

You may also run checks on their own with the appropriate command:

```bash
# Docstring linting
poetry run pydoclint src

# Linting (with automatic fixes)
poetry run ruff check --fix

# Check for security issues
poetry run bandit -r src --severity-level medium --confidence-level medium --quiet

# Run tests
poetry run pytest
```

Pushing directly to main is prohibited. If you wish your work to be included, first push a separate branch to
the repository and then open a Pull Request. All branches that have not received a commit for more than 30 days
will be automatically deleted.

## Versioning and releases

Versioning follows a simple model featuring three integers known as the major version, minor version and build number.
For example, in package version "v3.5.7", the major version is 3, the minor version is 5 and the build number is 7.

- The Major version should change when the package is entirely or almost entirely re-written.
  This should happen every few years for a well-maintained package.
- The Minor version should change when new user-facing functionality is added to the package.
- The Build number should change whenever a bug is fixed or a small tweak is made to the package.

The build number of this package is automatically increased when new code is merged to main (i.e. when a Pull Request
is closed). We automatically tag the new version and generate a GitHub release for it.

Whenever you make a change a change that warrants a minor or major version jump, you must:

- Determine the new version - if you are updating the minor version, the major should remain unchanged, and the build
  number should go back to 0 (e.g. v1.0.54 -> v1.1.0). If you are updating the major version, both the minor and the build
  should be decreased to 0 (e.g. v1.1.43 ->v2.0.0).
- Update the version in [tool.poetry] section of the pyproject.toml file.
- Add a line in CHANGELOG.md (at the top of the file, so that reading down the file has the reader moving backwards in
  time), updating the version number and describing what has been changed in the new major or minor version.
