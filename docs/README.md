# LabOne Q documentation

This subdirectory contains the documentation for LabOneQ.

The last released version can be found here: https://docs.zhinst.com/labone_q_user_manual

## Development

The Documentation is generated with [MkDocs](https://www.mkdocs.org/) and 
[Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme
to generate the documentation.

### Build the documentation locally

MkDocs is build with python, making it easy to install all requirement with `pip`.
The documentation therefore has its own `requirements.txt` 

Inside the `docs` directory, execute:

```
pip install requirements.txt
```

After this you should be able to build the documentation with:

```
mkdocs build
```

MkDocs also comes with a handy live preview that automatically adapts to your
local changes.

```
mkdocs serve
```

### Structure
The documentation is structured in four parts: 

* `content/`: contains all markdown source files
* `mkdocs.yml`: Configuration file for MKDocs. __INCLUDING__ the navigation content
* `overrides`: Theme customization
* `gen_file/`: Python script that are used in combination with the
    [gen-files](https://oprypin.github.io/mkdocs-gen-files/) plugin

### Automatic package reference documentation

The package documentation is currently divided into two parts.

[mkdocstrings](https://mkdocstrings.github.io/) is used to automatically
generate the documentation for each element.

[gen-files](https://oprypin.github.io/mkdocs-gen-files/) is used to automatically
create the markdown files with the mkdocstrings identifiers.
`docs/gen-files/reference_doc.py` is used as a prebuild script for gen-files.

To keep the live preview responsive mkdocstrings is disabled by default. 
The environment variable MKDOCSTRINGS, which takes a boolean value, can
be used to enable the plugin. 

Linux/Mac

```
(export MKDOCSTRINGS=true; mkdocs serve)
```

Windows

```
set "MKDOCSTRINGS=true" && mkdocs serve
```

### Jupyter notebook integration

[mkdocs-jupyter](https://github.com/danielfrg/mkdocs-jupyter) is used to
include jupyter notebooks into the documentation.

In order to avoid copying all notebooks into the content directory, a simple 
prebuild script `docs/gen-files/external_link.py` is used. It allows having
placeholders for additional files that automatically get added during the 
build step without the need of a symbolic link.

### Additional notes

Currently, we use
[`use_directory_urls= True`](https://www.mkdocs.org/user-guide/configuration/#use_directory_urls)m
which is the default. This means that if one views the static build sources in
a browser, links will not work properly. 
To view the static build sources locally, one needs to spin up a web server.
Thankfully, this is quite easy in python:

In the terminal, simply execute
```
python -m http.server -d path/to/static/sources
```

 