# FEniCSx 4 Flow - demo simulation

Example simulation as part of the FEniCS 4 Flow project, illustrating logging, plotting and post processing in the FX4F ecosystem. Meant to interface with the containarization support from [fx4f_runner](https://gitlab.com/fenicsx4flow/fx4f_runner). This project can be forked, and used as the basis for new projects.

## Running

Primarily, simulations should be run through the [fx4f_runner](https://gitlab.com/fenicsx4flow/fx4f_runner) supporting framework. Sometimes, it may be useful to execute the ``simulation.py`` file mode directly (e.g. for debugging purposes). To do so, run:

```
docker run --volume="$(pwd):/app" --user $(id -u):$(id -g) steinkfstoter/fx4f:latest python3 simulation.py <arguments>
```

or, if you want interactive plotting:

```
docker run --net=host --env="DISPLAY" --env="QT_X11_NO_MITSHM=1" --env="PYVISTA_OFF_SCREEN=false" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --volume="$HOME/.Xauthority:/home/developer/.Xauthority:rw" \
        --volume="$(pwd):/app" --user $(id -u):$(id -g) \
        steinkfstoter/fx4f:latest python3 simulation.py --showplots True <arguments>
```

where `<arguments>` can be specified as (e.g.) `--nx 16 --ny 16 --T 5`, etc.

The docker image `steinkfstoter/fx4f:latest` can also be build locally from [fx4f_runner](https://gitlab.com/fenicsx4flow/fx4f_runner).

## Development in VSCode

The above containerized system can be used also in your VSCode development environment, enabling the full support of the IDE's IntelliSys/auto-complete/linting/etc capabilities and of VSCode's debugger, irrespective of your local installation (which may not even house FEniCSx). This is achieved through the settings file in the `.devcontainer` folder. To make use of this set-up, you'll need to install the `dev container` VSCode extension. Upon opening VSCode, this should create a pop-up window with the button `reopen in container`. On Windows through WSL, you'll need to activate the Dev Containers setting `Execute in WSL`. Due to the installed python VSCode extensions in the dev container, running and debugging works out of the box (`F5` and debugger button). Further VSCode specification such as dedicated debug behavior, creating multiple run tasks with specific environment variables and/or arguments, etc., can be achieved by specification in `launch.json` (for debugging) and `tasks.json` (for running) files in a `.vscode` folder.