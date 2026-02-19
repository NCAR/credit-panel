import panel as pn
import param
import subprocess

class CommandRunner(param.Parameterized):
    command_input = param.String(default="")
    output_log = param.String(default="Terminal ready...")

    def __init__(self, **params):
        super().__init__(**params)
        
        # 1. Text Input
        self.editor = pn.widgets.TextInput(
            name="Command input",
            placeholder="Type and press Enter...",
            sizing_mode='stretch_width',
            value=self.command_input
        )
        self.editor.link(self, value='command_input')
        self.editor.param.watch(self._execute, 'value')
        
        # 2. Button - Added 'visible=True' and explicit height to prevent 0px rendering
        self.run_btn = pn.widgets.Button(
            name="Run Command", 
            button_type="primary",
            height=40,
            min_width=120,
            visible=True,
            sizing_mode='fixed'
        )
        self.run_btn.on_click(self._execute)
        
        # 3. Output area
        self.console = pn.widgets.StaticText(
            name="Output log",
            value=f"<pre style='background:#f4f4f4; padding:5px;'>{self.output_log}</pre>",
            sizing_mode='stretch_width'
        )

    def _execute(self, event):
        cmd = self.editor.value.strip()
        if not cmd: return

        try:
            # Executes in the active conda environment
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            response = result.stdout if result.returncode == 0 else result.stderr
            self.output_log = response if response else "Done."
        except Exception as e:
            self.output_log = str(e)
            
        self.console.value = f"<pre style='background:#f4f4f4; padding:5px;'>{self.output_log}</pre>"

    def panel(self):
        """Returns a Column with explicit sizing to force visibility."""
        return pn.Column(
            "### CommandRunner",
            self.editor,
            self.run_btn, # Explicitly included in the column
            self.console,
            sizing_mode='stretch_width',
            min_height=300 # Forces the container to expand
        )

# --- DEBUG TEST ---
# Run this block directly to see if the button appears in isolation
#runner = CommandRunner()
#runner.panel().servable()
