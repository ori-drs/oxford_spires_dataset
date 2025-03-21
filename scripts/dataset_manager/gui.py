import asyncio
from pathlib import Path

from dataset_downloader import DatasetDownloader
from dir_utils import get_common_directories
from nicegui import ui


class OxSpiresGUI:
    def __init__(self):
        self.downloader = DatasetDownloader()
        self.status_label = None
        self.dir_select = None
        self.sequence_status_list = None

        # Create the main application UI with better centering
        with ui.row().classes("w-full justify-center"):
            ui.label("Oxford Spires Dataset").classes("text-h4 q-my-md")
        # Create tabs
        with ui.tabs().classes("w-full justify-center") as tabs:
            ui.tab("Dataset Manager", icon="cloud_download")
            ui.tab("Visualization", icon="bar_chart")
            ui.tab("Analysis", icon="analytics")

        # Create tab panels that are linked to the tabs
        with ui.tab_panels(tabs, value="Dataset Manager").classes("w-full"):
            with ui.tab_panel("Dataset Manager"):
                self.create_dataset_manager_tab()

            with ui.tab_panel("Visualization"):
                self.create_visualization_tab()

            with ui.tab_panel("Analysis"):
                self.create_analysis_tab()

        ui.run(host="0.0.0.0", port=8080)

    def update_directory_status(self):
        # global status_label, sequence_status_list
        """Update the status information for the current directory."""

        # If status_label hasn't been created yet, return early
        if self.status_label is None:
            print("status_label is None")
            return
        if self.sequence_status_list is None:
            print("sequence_status_list is None")
            return

        base_dir = Path(self.downloader.base_dir)
        if not base_dir.exists():
            self.status_label.text = f"Directory does not exist: {base_dir}"
            return

        # Clear existing rows in the table
        self.sequence_status_list.clear()
        # Refresh local sequences
        self.downloader.load_local_sequences()

        # Add rows for each remote sequence
        for sequence in self.downloader.local_sequences:
            status = self.downloader.get_local_sequence_status(sequence)

            # Create status indicators
            {
                "available": "✅ Complete",
                "incomplete": "⚠️ Incomplete",
                "not_found": "❌ Not Downloaded",
                "invalid": "❓ Invalid",
            }.get(status, "❓ Unknown")

            # Add the row to the table
            # row = {"sequence": sequence, "status": status_icon}
            # status_table.add_row(row)

            with self.sequence_status_list:
                with ui.expansion("Expand!", icon="check").classes("w-full"):
                    ui.label("inside the expansion")
                with ui.expansion("Incomplete!", icon="warning").classes("w-full"):
                    ui.label("inside the expansion")
        # Update the status label with the current directory information
        status_text = f"Current directory: {base_dir}\n"
        self.status_label.text = status_text

    async def handle_directory_change(self, e):
        """Handle directory selection or input change."""

        new_dir = Path(self.dir_select.value)

        # Check if directory exists
        if not new_dir.exists():
            # Ask user if they want to create the directory
            if await ui.dialog(
                "Directory does not exist", f'The directory "{new_dir}" does not exist. Would you like to create it?'
            ).confirm():
                try:
                    new_dir.mkdir(parents=True, exist_ok=True)
                    ui.notify(f"Created directory: {new_dir}", type="positive")
                except Exception as ex:
                    ui.notify(f"Failed to create directory: {str(ex)}", type="negative")
                    self.dir_select.value = str(self.downloader.base_dir)
                    return
            else:
                # If user doesn't want to create it, revert to previous directory
                self.dir_select.value = str(self.downloader.base_dir)
                return

        # Update the downloader's base directory
        self.downloader.base_dir = new_dir  # Pass Path object directly instead of string
        self.downloader.load_local_sequences()  # Refresh local sequences list
        self.update_directory_status()

    def create_dataset_manager_tab(
        self,
    ):
        async def download_sequence():
            if not sequence_select.value:
                ui.notify("Please select a sequence", type="warning")
                return

            # Check if directory exists before downloading
            if not Path(self.downloader.base_dir).exists():
                ui.notify("Selected directory does not exist. Please select a valid directory.", type="negative")
                return

            progress.visible = True
            self.status_label.text = f"Downloading sequence: {sequence_select.value}"

            # Use download_subfolder instead of download_sequence
            success = await asyncio.to_thread(self.downloader.download_subfolder, f"{sequence_select.value}/raw")

            if success:
                ui.notify("Sequence downloaded successfully!", type="positive")
                self.downloader.load_local_sequences()  # Refresh local sequences
                self.update_directory_status()  # Update status after successful download
            else:
                ui.notify("Failed to download sequence", type="negative")

            progress.visible = False

        async def download_ground_truth():
            if not site_select.value:
                ui.notify("Please select a site", type="warning")
                return

            # Check if directory exists before downloading
            if not Path(self.downloader.base_dir).exists():
                ui.notify("Selected directory does not exist. Please select a valid directory.", type="negative")
                return

            progress.visible = True
            self.status_label.text = f"Downloading ground truth for: {site_select.value}"

            success = await asyncio.to_thread(self.downloader.download_ground_truth, site_select.value)

            if success:
                ui.notify("Ground truth downloaded successfully!", type="positive")
                self.update_directory_status()  # Update status after successful download
            else:
                ui.notify("Failed to download ground truth", type="negative")

            progress.visible = False

        # Pre-declare variables that will be used in the nested functions
        sequence_select = None
        site_select = None
        progress = None

        # Now build the UI
        with ui.column().classes("max-w-2xl mx-auto p-4"):
            # Directory selection section
            ui.label("Select Base Directory").classes("text-h6 q-mb-md text-center")
            with ui.row().classes("justify-center"):
                self.dir_select = ui.select(options=get_common_directories(), label="Select Directory", with_input=True)
                self.dir_select.props('style="width: 300px"')
                self.dir_select.value = str(self.downloader.base_dir)
                self.dir_select.on("update:model-value", self.handle_directory_change)

            # Status label
            self.status_label = ui.label("").classes("text-center q-mt-md q-mb-md whitespace-pre-line")

            # Create a table to display the status of sequences
            self.sequence_status_list = ui.card().classes("w-full q-mt-md")
            print("sequence_status_list", self.sequence_status_list)

            self.update_directory_status()

            # Sequence download section
            ui.label("Download Sequence").classes("text-h6 q-mb-md text-center")
            with ui.row().classes("justify-center"):
                sequence_select = ui.select(
                    options=self.downloader.remote_sequences, label="Select Sequence", with_input=True
                )
                sequence_select.props('style="width: 300px"')

            # Add download sequence button here
            with ui.row().classes("justify-center q-mb-lg"):
                ui.button("Download Sequence", on_click=download_sequence).props("color=primary")

            # Ground truth download section
            ui.label("Download Ground Truth").classes("text-h6 q-mb-md text-center")
            with ui.row().classes("justify-center"):
                site_select = ui.select(
                    options=self.downloader.list_available_sites(), label="Select Site", with_input=True
                )
                site_select.props('style="width: 300px"')

            # Progress bar
            progress = ui.linear_progress(0).classes("w-full")
            progress.visible = False

            # Only ground truth download button here
            with ui.row().classes("justify-center gap-4"):
                ui.button("Download Ground Truth", on_click=download_ground_truth).props("color=secondary")

        # Call the function to update the directory status
        self.update_directory_status()

    def create_visualization_tab(
        self,
    ):
        """Create the Visualization tab."""
        with ui.column().classes("max-w-2xl mx-auto p-4"):
            ui.label("Data Visualization").classes("text-h4 q-mb-md text-center")
            ui.label("Visualization tools will be implemented here.").classes("text-center")

            # Placeholder for visualization controls
            with ui.card().classes("w-full"):
                ui.label("Visualization Controls").classes("text-h6")
                ui.label("Select visualization options:").classes("q-mb-md")

                with ui.row():
                    ui.checkbox("Show trajectories")
                    ui.checkbox("Show landmarks")

                ui.button("Generate Visualization", color="primary").classes("q-mt-md")

    def create_analysis_tab(self):
        """Create the Analysis tab."""
        with ui.column().classes("max-w-2xl mx-auto p-4"):
            ui.label("Data Analysis").classes("text-h4 q-mb-md text-center")
            ui.label("Analysis tools will be implemented here.").classes("text-center")

            # Placeholder for analysis controls
            with ui.card().classes("w-full"):
                ui.label("Analysis Parameters").classes("text-h6")

                with ui.row().classes("q-mb-md"):
                    ui.select(["Option 1", "Option 2", "Option 3"], label="Analysis Type").classes("w-full")

                with ui.row():
                    ui.number("Threshold", value=0.5, min=0, max=1, step=0.1)

                ui.button("Run Analysis", color="primary").classes("q-mt-md")


if __name__ in {"__main__", "__mp_main__"}:
    gui = OxSpiresGUI()
