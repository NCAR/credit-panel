import pandas as pd
import panel as pn
import param

# Initialize the extension (standard for Jupyter/Casper environments)
pn.extension()

class DatasetBrowser(param.Parameterized):
    # Track multi-select state
    checked_items = param.List(default=[])
    # Track single "active" focus state
    active_dataset = param.String(default="")

    def __init__(self, datasets, **params):
        super().__init__(**params)
        self.datasets = datasets
        print(self.datasets)
        # Storage for row objects to allow dynamic style updates
        self._rows = {}

    def _get_row_style(self, name):
        """Calculates the CSS for a row based on whether it is active."""
        is_active = (name == self.active_dataset)
        return {
            'background': '#e8f2ff' if is_active else 'transparent',
            'border-left': '5px solid #007bff' if is_active else '5px solid transparent',
            'display': 'flex',
            'align-items': 'center',
            'width': '100%',
            'border-radius': '0 4px 4px 0',
            'transition': 'background 0.2s', # Smooth color transition
            'padding' : '0px 2px',
            'margin' : '0pz'
        }

    def _set_active(self, name):
        """Centralized logic to move the highlight and refresh the UI."""
        self.active_dataset = name
        self._update_ui()

    def _make_row(self, name):
        # 1. Checkbox for multi-select
        # align='center' and specific margins ensure vertical centering with text
        cb = pn.widgets.Checkbox(
            name="", value=False, width=10, 
            align='center', margin=(10, 0, 2, 0) 
        )
        
        def update_checked(event):
            current = list(self.checked_items)
            if event.new:
                if name not in current: current.append(name)
                # Highlight the row when the box is checked
                self._set_active(name)
            else:
                if name in current: current.remove(name)
            self.checked_items = current
            
        cb.param.watch(update_checked, 'value')

        # 2. Button styled as a flat label for single-select/highlight
        # align='center' and line-height: 30px match the checkbox height
        btn = pn.widgets.Button(
            name=name, 
            button_type='default', 
            sizing_mode='stretch_width',
            align='center',
            stylesheets=["""
                .bk-btn {
                    background: transparent !important;
                    border: none !important;
                    box-shadow: none !important;
                    text-align: left !important;
                    padding-left: 5px !important;
                    line-height: 6px !important;
                    font-size: 14px !important;
                    color: #333 !important;
                    cursor: pointer;
                    outline: none !important;
                    box-shadow: none !important;
                }
                .bk-btn:hover { 
                    background: rgba(0,0,0,0.05) !important; 
                }
            """]
        )
        
        # Highlight row when the dataset name is clicked
        btn.on_click(lambda event: self._set_active(name))

        # 3. Create the Row container
        row = pn.Row(
            cb, 
            btn, 
            styles=self._get_row_style(name), 
            sizing_mode='stretch_width',
            margin=0
        )
        
        self._rows[name] = row
        return row

    def _update_ui(self):
        """Forces the CSS styles of all rows to refresh."""
        for name, row in self._rows.items():
            row.styles = self._get_row_style(name)
            # Explicitly trigger the parameter update for older Panel versions
            row.param.trigger('styles')

    @property
    def panel(self):
        """Returns the scrollable list of datasets."""
        return pn.Column(
            *[self._make_row(d) for d in self.datasets],
            sizing_mode='stretch_width',
            max_height=600,
            scroll=True,
            styles={'border': '1px solid #ddd', 'border-radius': '4px', 'background': 'white'}
        )

# --- App Construction ---

## Simulated Dataset Names
#dataset_names = [
#    "2026-02-18T00:00Z_Control",
#    "2026-02-18T06:00Z_Control",
#    "2026-02-18T12:00Z_Exp_v1",
#    "2026-02-18T18:00Z_Exp_v1",
#    "2026-02-19T00:00Z_Exp_v2",
#    "Dataset_Ref_Standard",
#    "Calibration_Run_01"
#]
#
## Instantiate the browser
#browser = DatasetBrowser(datasets=dataset_names)
#
## Define a display area to show what is happening (for debugging)
#status_info = pn.Column(
#    pn.bind(lambda x: f"### 🔍 Focused Dataset: `{x}`" if x else "### 🔍 Select a dataset", browser.param.active_dataset),
#    pn.bind(lambda x: f"**✅ Items to Plot:** {', '.join(x) if x else 'None'}", browser.param.checked_items),
#    width=400,
#    margin=(0, 0, 0, 20)
#)
#
## Final Layout
#layout = pn.Row(
#    pn.Column("## Dataset Browser", browser.panel, width=350),
#    status_info
#).servable()
