# Panel App – How to Run on NCAR JupyterHub (HPC)
#
# 1) Log into JupyterHub:
#    https://jupyterhub.hpc.ucar.edu/
#
# 2) Open a Terminal in JupyterLab
#
# 3) Activate the conda environment (if needed):
#    conda activate <YOUR_CONDA_ENV>
#
# 4) Navigate to this file’s directory
#
# 5) Start the Panel server (keep this terminal running):
#    panel serve panel_app.py --address 127.0.0.1 --port 5006 \
#        --allow-websocket-origin="jupyterhub.hpc.ucar.edu"
#
# 6) Open the app in your browser (same JupyterHub session):
#    https://jupyterhub.hpc.ucar.edu/stable/user/<USER_NAME>/proxy/5006/panel_app
#
# 7) Stop the app:
#    Go back to the terminal and press Ctrl+C

import xarray as xr
from pathlib import Path
from era5_plot import plot_png, NETCDF_FILE, VAR_NAME, TIME_NAME, LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME
import panel as pn
import param

from datasetSelector2 import DatasetBrowser
from metadata import DatasetMetadata
from datasetPlot import DatasetPlot2
from commandRunner import CommandRunner

pn.extension(raw_css=[Path("static/styles.css").read_text()])

DATA_DIR = Path("/Users/vapor/Data/model_predict")
DATASET_METADATA = {}

def scan_datasets():
    for d in DATA_DIR.iterdir():
        if d.is_dir():
            nc_file = f"{d}/*.nc"
            with xr.open_mfdataset(nc_file, engine="netcdf4", autoclose=True) as ds:
                DATASET_METADATA[d.name] = {
                    "ntime": len(ds.time),
                    "nlev": len(ds.get(LEV_NAME, [])),
                    "nplev": int(ds.sizes[PRES_NAME]),
                    "nlat": int(ds.sizes[LAT_NAME]),
                    "nlon": int(ds.sizes[LON_NAME]),
                    "stime": str(ds.time.values[0].astype("datetime64[s]")),
                    "etime": str(ds.time.values[-1].astype("datetime64[s]")),
                    "vars2d": [v for v in ds.data_vars if len(ds[v].dims) <= 3],
                    "vars3d": [v for v in ds.data_vars if len(ds[v].dims) > 3]
                }

scan_datasets()

def available_datasets():
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir()
    )

browser = DatasetBrowser(datasets=available_datasets())

@pn.depends(browser.param.checked_items)
def plot_grid(datasets):

    if not datasets:
        return pn.pane.Markdown("### Select one or more datasets")

    plots = [
        DatasetPlot2(dataset=ds, metadata=DATASET_METADATA).panel()
        for ds in datasets
    ]

    return pn.GridBox(
        *plots,
        ncols=2,
        sizing_mode=None,
        css_classes=["plot-grid"],
        styles={
            "grid-auto-rows": "min-content",
            "align-items": "start"
        },
    )

metadata = DatasetMetadata(metadata=DATASET_METADATA)
def sync_active_dataset(event):
    metadata.active_key = event.new
browser.param.watch(sync_active_dataset, 'active_dataset')

sidebar = pn.Column(
    "## Datasets",
    browser.panel,
    metadata.panel,
    width=250
)

main = pn.Column(
    plot_grid,
    sizing_mode="stretch_width",
    css_classes=["main-content"]    
)

vis = pn.Row(
    sidebar,
    main,
    sizing_mode="stretch_both",
    styles={"height" : "100vh"}
)

commandRunner = CommandRunner()
inference = pn.Column(
    #pn.widgets.TextEditor(placeholder='Enter some text'),
    commandRunner.panel()
)

tabs = pn.Tabs(
    ("Visualization", vis),
    ("Inference", inference)
).servable()
