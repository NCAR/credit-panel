# Step 1: Load datasets dynamically
import xarray as xr
from pathlib import Path
from era5_plot import plot_png, NETCDF_FILE, VAR_NAME, TIME_NAME, LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME
import panel as pn
import param

from datasetSelector2 import DatasetBrowser
from metadata import DatasetMetadata
from datasetPlot import DatasetPlot2

#pn.extension(css_files=["static/styles.css"])
pn.extension(raw_css=[Path("static/styles.css").read_text()])
#pn.extension()
#pn.config.css_files.append("static/styles.css")

DATA_DIR = Path("/Users/vapor/Data/model_predict")
DATASET_METADATA = {}

def scan_datasets():
    #print(DATA_DIR)
    for d in DATA_DIR.iterdir():
        #print(d)
        if d.is_dir():
            #nc_file = d / "*.nc" # Assuming standard naming
            nc_file = f"{d}/*.nc"
            #if nc_file.exists():
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
#print(DATASET_METADATA)

def available_datasets():
    return sorted(
        d.name for d in DATA_DIR.iterdir()
        if d.is_dir()
    )

browser = DatasetBrowser(datasets=available_datasets())

# Step 3: Build the image grid reactively
#@pn.depends(dataset_select.param.value)
@pn.depends(browser.param.checked_items)
def plot_grid(datasets):

    if not datasets:
        return pn.pane.Markdown("### Select one or more datasets")

    plots = [
        #DatasetPlot(dataset=ds).panel()
        DatasetPlot2(dataset=ds, metadata=DATASET_METADATA).panel()
        for ds in datasets
    ]

    return pn.GridBox(
        *plots,
        ncols=2,
        #sizing_mode="stretch_width",
        sizing_mode=None,
        css_classes=["plot-grid"],
        styles={
            "grid-auto-rows": "min-content",
            "align-items": "start"
        },
        #styles={"gap"        : "15px",
        #        "margin-top" : "20px"
        #}
    )

metadata = DatasetMetadata(metadata=DATASET_METADATA)
# Instead of browser.link(metadata_viewer, active_dataset='active_key')
def sync_active_dataset(event):
    metadata.active_key = event.new
browser.param.watch(sync_active_dataset, 'active_dataset')

print(DATASET_METADATA)

# Layout
sidebar = pn.Column(
    "## Datasets",
    #dataset_select,
    browser.panel,
    metadata.panel,
    width=250
    #width=550
    #css_classes=["credit-app"]    
    #styles={"background"   : "#f5f5f5",
    #        "padding"      : "15px",
    #        "border-right" : "1px solid #ccc",
    #        "text-align"   : "left",
    #        "overflow-y"   : "auto"
    #}
)

main = pn.Column(
    plot_grid,
    #sizing_mode="stretch_both",
    sizing_mode="stretch_width",
    css_classes=["main-content"]    
)

vis = pn.Row(
    sidebar,
    main,
    sizing_mode="stretch_both",
    styles={"height" : "100vh"}
    #css_classes=["credit-app"]    
)
#).servable()

inference = pn.Column(
    pn.widgets.TextEditor(placeholder='Enter some text')
)

tabs = pn.Tabs(
    ("Visualization", vis),
    ("Inference", inference)
).servable()
