# dataset_metadata.py
import panel as pn
import param

class DatasetMetadata(param.Parameterized):
    metadata = param.Dict(default={})
    active_key = param.String(default="")

    @pn.depends('active_key', 'metadata')
    def panel(self):
        if not self.active_key or self.active_key not in self.metadata:
            return pn.pane.HTML("<div style='padding:10px; font-family: sans-serif;'><i>Select a dataset</i></div>")

        data = self.metadata[self.active_key]
        v2d = ", ".join(data.get('vars2d', []))
        v3d = ", ".join(data.get('vars3d', []))

        html_content = f"""
        <style>
            .metadata-container {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                font-size: 12px;
                padding: 5px 10px;
                background-color: white;
                overflow-wrap: break-word; /* Prevents text from exceeding width */
            }}
            .meta-row {{
                display: flex;
                justify-content: space-between;
                padding: 2px 0;
            }}
            .meta-label {{
                font-weight: bold;
                color: #444;
                min-width: 90px;
            }}
            .meta-value {{
                text-align: right;
                color: #222;
                flex-grow: 1;
            }}
            .var-section {{
                margin-top: 10px;
                border-top: 1px solid #eee;
                padding-top: 8px;
            }}
            .var-row {{
                display: flex;
                margin-bottom: 4px;
            }}
            .var-list {{
                color: #666;
                padding-left: 10px;
                line-height: 1.4;
            }}
        </style>
        <div class="metadata-container">
            <div class="meta-row"><span class="meta-label">Name:</span><span class="meta-value">{self.active_key}</span></div>
            <div class="meta-row"><span class="meta-label">Ts Start:</span><span class="meta-value">{data.get('stime', 'N/A')}</span></div>
            <div class="meta-row"><span class="meta-label">Ts End:</span><span class="meta-value">{data.get('etime', 'N/A')}</span></div>
            <div class="meta-row"><span class="meta-label">Latitudes:</span><span class="meta-value">{data.get('nlat', 'N/A')}</span></div>
            <div class="meta-row"><span class="meta-label">Longitudes:</span><span class="meta-value">{data.get('nlon', 'N/A')}</span></div>
            <div class="meta-row"><span class="meta-label">Levels:</span><span class="meta-value">{data.get('nlev', 'N/A')}</span></div>
            <div class="meta-row"><span class="meta-label">P-Levels:</span><span class="meta-value">{data.get('nplev', 'N/A')}</span></div>
            <div class="meta-row"><span class="meta-label">Forecasts:</span><span class="meta-value">{data.get('ntime', 'N/A')}</span></div>
            
            <div class="var-section">
                <div class="var-row"><span class="meta-label">Vars 2D:</span><span class="var-list">{v2d}</span></div>
                <div class="var-row"><span class="meta-label">Vars 3D:</span><span class="var-list">{v3d}</span></div>
            </div>
        </div>
        """
        # Ensure the pane itself doesn't force a horizontal scroll
        data = pn.pane.HTML(html_content, sizing_mode='stretch_width')
        return pn.Column("## Metadata", data)
