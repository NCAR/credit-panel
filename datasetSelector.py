import panel as pn
import param

class DatasetBrowser1(param.Parameterized):
    # List of names currently checked
    checked_datasets = param.List(default=[])
    
    # The single dataset currently "Active" (Highlighted)
    active_dataset = param.String(default="")

    def __init__(self, datasets, **params):
        super().__init__(**params)
        self.datasets = datasets
        self._rows = {} # Cache for UI updates

    def _get_row_style(self, name):
        """Dynamic styling: Highlight if active, otherwise transparent."""
        is_active = (name == self.active_dataset)
        return {
            "background": "#e8f2ff" if is_active else "transparent",
            "border-left": "4px solid #007bff" if is_active else "4px solid transparent",
            "padding": "5px",
            "margin": "2px 0",
            "border-radius": "0 4px 4px 0",
            "cursor": "pointer",
            "display": "flex",
            "align-items": "center"
        }

    def _make_row(self, name):
        # 1. Multi-select Checkbox
        cb = pn.widgets.Checkbox(name="", value=(name in self.checked_datasets), width=25)
        
        @pn.depends(cb.param.value, watch=True)
        def sync_checked(val):
            current = list(self.checked_datasets)
            if val and name not in current:
                current.append(name)
            elif not val and name in current:
                current.remove(name)
            self.checked_datasets = current

        # 2. Clickable Label (StaticText + transparent overlay style)
        label = pn.widgets.StaticText(value=name, align="center", margin=(0,0,0,10))
        
        # 3. Create Row and make it interactive
        # We use a button-like interaction by wrapping in a Clickable Column if Row fails
        row = pn.Row(cb, label, styles=self._get_row_style(name), sizing_mode="stretch_width")
        
        # Link the click to the active selection
        def set_active(event):
            self.active_dataset = name
            self.update_styles()
            
        # Using on_click on a Layout requires 'pointer-events' to be active
        row.on_click(set_active)
        
        self._rows[name] = row
        return row

    def update_styles(self):
        """Refresh background colors when selection changes."""
        for name, row in self._rows.items():
            row.styles = self._get_row_style(name)

    @property
    def panel(self):
        return pn.Column(
            *[self._make_row(d) for d in self.datasets],
            sizing_mode="stretch_width"
        )

import panel as pn
import param

class DatasetBrowser(param.Parameterized):
    checked_datasets = param.List(default=[])
    active_dataset = param.String(default="")

    def __init__(self, datasets, **params):
        super().__init__(**params)
        self.datasets = datasets
        self._rows = {} 

    def _get_row_style(self, name):
        is_active = (name == self.active_dataset)
        return {
            "background": "#e8f2ff" if is_active else "transparent",
            "border-left": "4px solid #007bff" if is_active else "4px solid transparent",
            "padding": "2px",
            "margin": "0",
            "display": "flex",
            "align-items": "center",
            "width": "100%"
        }

    def _make_row(self, name):
        # 1. Multi-select Checkbox
        cb = pn.widgets.Checkbox(name="", value=(name in self.checked_datasets), width=25)
        
        def sync_checked(event):
            current = list(self.checked_datasets)
            if event.new and name not in current:
                current.append(name)
            elif not event.new and name in current:
                current.remove(name)
            self.checked_datasets = current
        cb.param.watch(sync_checked, 'value')

        # 2. Clickable Label using a Button with custom CSS
        # This replaces row.on_click to avoid AttributeError
        btn = pn.widgets.Button(
            name=name,
            button_type="default",
            sizing_mode="stretch_width",
            stylesheets=["""
                :host(.bk-btn-default) .bk-btn {
                    background-color: transparent !important;
                    border: none !important;
                    text-align: left !important;
                    box-shadow: none !important;
                    padding: 8px 5px !important;
                    font-size: 14px;
                    cursor: pointer;
                }
                :host(.bk-btn-default) .bk-btn:hover {
                    background-color: rgba(0,0,0,0.03) !important;
                }
            """]
        )
        
        def set_active(event):
            self.active_dataset = name
            self.update_styles()
        
        btn.on_click(set_active)

        # 3. Create Row container
        row = pn.Row(cb, btn, styles=self._get_row_style(name), sizing_mode="stretch_width")
        
        self._rows[name] = row
        return row

    def update_styles(self):
        for name, row in self._rows.items():
            row.styles = self._get_row_style(name)

    @property
    def panel(self):
        return pn.Column(
            *[self._make_row(d) for d in self.datasets],
            sizing_mode="stretch_width"
        )

# Example setup
ds_list = ["2026-01-26T00Z", "2026-01-28T06Z", "Model_v1", "Model_v2_Beta"]
browser = DatasetBrowser(datasets=ds_list)

sidebar = pn.Column(
    "## Datasets",
    browser.panel,
    width=280
)

# Bind the main view to the active dataset
main_view = pn.Column(
    pn.bind(lambda ds: f"# Active Dataset: {ds}" if ds else "# Select a Dataset", browser.param.active_dataset),
    pn.bind(lambda checked: f"**Currently Plotting:** {', '.join(checked)}", browser.param.checked_datasets)
)

pn.Row(sidebar, main_view).servable()
