import panel as pn
import os

pn.extension()

class RemoteDirPicker:
    def __init__(self, start_path="."):
        self.current_path = pn.widgets.TextInput(
            name="Current Path", value=os.path.abspath(start_path)
        )

        self.dir_list = pn.widgets.Select(size=12)
        self.select_button = pn.widgets.Button(name="Select", button_type="primary")

        self._callback = None

        self.dir_list.param.watch(self._navigate, "value")
        self.select_button.on_click(self._select)

        self._refresh()

    def _refresh(self):
        try:
            entries = os.listdir(self.current_path.value)
            dirs = sorted(
                d for d in entries
                if os.path.isdir(os.path.join(self.current_path.value, d))
            )
            self.dir_list.options = [".."] + dirs
        except Exception as e:
            self.dir_list.options = [str(e)]

    def _navigate(self, event):
        if event.new == "..":
            self.current_path.value = os.path.dirname(self.current_path.value)
        else:
            new_path = os.path.join(self.current_path.value, event.new)
            if os.path.isdir(new_path):
                self.current_path.value = os.path.abspath(new_path)
        self._refresh()

    def _select(self, _):
        if self._callback:
            self._callback(self.current_path.value)

    def open(self, callback):
        """Open picker and register callback"""
        self._callback = callback
        return pn.Column(
            "### Select Directory",
            self.current_path,
            self.dir_list,
            self.select_button
        )


# --- Main UI ---

#selected_path = pn.widgets.TextInput(name="Selected Directory")
#
#button = pn.widgets.Button(name="Browse", button_type="primary")
#
#def open_picker(event):
#    picker = RemoteDirPicker()
#
#    def on_select(path):
#        selected_path.value = path
#        pn.state.modal.close()
#
#    pn.state.modal.open(picker.open(on_select))
#
#button.on_click(open_picker)
#
#pn.Column(
#    pn.Row(selected_path, button)
#).servable()
