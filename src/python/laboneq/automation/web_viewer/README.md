# Automation Web Viewer

Interactive D3.js-based visualization of the automation graph.

## Vendored Dependencies

The web viewer bundles a minified copy of [D3.js](https://d3js.org/) so that
it can run fully offline. The vendored file lives at:

    app/static/js/d3.latest.min.js

The file includes the D3 license header prepended to the minified source.

### Updating D3.js

To update to the latest version, run the provided script from this directory:

    ./vendor_d3.sh

The script will:

1. Query the npm registry for the latest D3.js version.
2. Prompt you to confirm the download.
3. Download the minified build from cdnjs.
4. Prepend the D3 license header and write app/static/js/d3.latest.min.js.

After updating, verify the viewer works and commit the changed file.

## Running the Viewer

    from laboneq.automation.web_viewer import start_web_viewer

    viewer = start_web_viewer(automation, port=5000)
    # ...
    viewer.stop()
