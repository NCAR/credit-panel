# Step 1: Load datasets dynamically
import xarray as xr
from pathlib import Path
from era5_plot import plot_png, NETCDF_FILE, VAR_NAME, TIME_NAME, LEV_NAME, PRES_NAME, LAT_NAME, LON_NAME
import panel as pn
import param

pn.extension(raw_css=[Path("static/styles.css").read_text()])

class DatasetPlot2(param.Parameterized):
    dataset = param.String()
    dimension = param.String()
    var_name = param.String(default="t2m")
    time_index = param.Integer()
    level_index = param.Integer()
    metadata = param.Dict(default={})

    def __init__(self, **params):
        super().__init__(**params)

        #self.dsMeta = DATASET_METADATA[self.dataset]
        #self.metadata = 
        print("self.metadata")
        print(self.metadata)
        self.metadata = self.metadata[self.dataset]

        def get_dropdown_width(options):
            longest = max([len(str(opt)) for opt in options])
            return max(80, (longest * 9) + 40)

        dims = ["2D", "3D"]
        self.dim_selector = pn.widgets.Select(
            name="",
            options=dims,
            value="2D",
            min_width=50,
            max_width=100,
            sizing_mode="stretch_width"
        )
        self.dim_selector.link(self, value="dimension")

        #self.var_name = self.dsMeta["vars2d"][0]
        self.var_name = self.metadata["vars2d"][0]
        self.var_selector = pn.widgets.Select(
            name="",
            #options=self.dsMeta["vars2d"],
            options=self.metadata["vars2d"],
            value=self.var_name,
            #max_width=get_dropdown_width(self.dsMeta["vars2d"]+self.dsMeta["vars3d"]),
            max_width=get_dropdown_width(self.metadata["vars2d"]+self.metadata["vars3d"]),
            sizing_mode="stretch_width"
        )
        self.var_selector.link(self, value="var_name")

        self.var_row= pn.Row(
            pn.widgets.StaticText(value="<b>Dimension</b>", width=70, align="center"),
            self.dim_selector,
            pn.widgets.StaticText(value="<b>Variable</b>", width=70, align="center"),
            self.var_selector,
            align="center",
            sizing_mode="stretch_width",
            max_width=800, # SET THIS to match your Plot width
            css_classes=["widget-row"]
        )

        self.time_slider = pn.widgets.IntSlider(
            name="",
            start=0,
            #end=self.dsMeta["ntime"],
            end=self.metadata["ntime"],
            value=self.time_index,
            show_value=False,
            sizing_mode="stretch_width"
        )
        self.time_slider.link(self, value="time_index")

        self.level_slider = pn.widgets.IntSlider(
            name="",
            start=0,
            #end=self.dsMeta["nlev"],
            end=self.metadata["nlev"],
            value=self.level_index,
            show_value=False,
            sizing_mode="stretch_width"
        )
        self.level_slider.link(self, value="level_index")

        time_display = pn.pane.HTML(
            pn.bind(lambda v: f"<b>Time:</b> {v}", self.time_slider.param.value),
            styles={'line-height': '30px', 'font-size': '14px'},
            width=50, 
            margin=0
        )
        level_display = pn.pane.HTML(
            pn.bind(lambda v: f"<b>Level:</b> {v}", self.level_slider.param.value),
            styles={'line-height': '30px', 'font-size': '14px'},
            width=60, 
            margin=0
        )

        self.slider_row = pn.Row(
            time_display,
            self.time_slider,
            level_display,
            self.level_slider,
            align="center",
            sizing_mode="stretch_width",
            max_width=800, # SET THIS to match your Plot width
            css_classes=["widget-row"]
        )
        
    @param.depends('dimension', watch=True)
    def _update_variable_options(self):
        # 1. Determine which list to use from metadata
        if self.dimension == "2D":
            new_options = self.metadata["vars2d"]
            #new_options = self.dsMeta["vars2d"]
        else:
            new_options = self.metadata["vars3d"]
            #new_options = self.dsMeta["vars3d"]
        
        # 2. Update the widget's options
        self.var_selector.options = new_options
        
        # 3. Reset the value to the first item in the new list 
        # to prevent "Value not in options" errors
        if new_options:
            self.var_selector.value = new_options[0]

    @pn.depends("time_index", "level_index", "var_name")
    def view(self):
        buf = plot_png(
            dataset=self.dataset,
            t=self.time_index,
            lev=self.level_index,
            var_name=self.var_name
        )

        return pn.pane.PNG(
            buf,
            #sizing_mode="stretch_width",
            sizing_mode="scale_width",
            align="center",
            height=None,
            min_height=None,
            max_height=None
            #max_width=800,
            #height=450
            #css_classes=["plot-panel"]
        )

    def panel(self):
        return pn.Column(
            pn.pane.Markdown(f"### {self.dataset}"),
            self.var_row,
            self.slider_row,
            self.view,
            align="center",
            sizing_mode="stretch_width",
            height=None,
            min_height=None,
            max_height=None,
            css_classes=["plot-container"]
            #styles={"border" : "1px solid #ccc",
            #        "padding" : "12px",
            #        "border-radius" : "10px",
            #        "background" : "white",
            #        "text-align" : "center"
            #}
        )
