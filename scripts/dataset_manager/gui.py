import asyncio
import os
from pathlib import Path

from dataset_downloader import DatasetDownloader
from nicegui import ui

# Initialize the dataset downloader
downloader = DatasetDownloader()

# Define status_label as a global variable
status_label = None
dir_select = None
status_table = None


def get_common_directories():
    """Get a list of common directories to choose from."""
    common_dirs = [
        str(Path.home()),
        str(Path.home() / "data"),
        str(Path.home() / "workspace"),
    ]
    return [d for d in common_dirs if os.path.exists(d)]


def update_directory_status():
    """Update the status information for the current directory."""

    # If status_label hasn't been created yet, return early
    if status_label is None:
        return

    base_dir = Path(downloader.base_dir)
    if not base_dir.exists():
        status_label.text = f"Directory does not exist: {base_dir}"
        return

    # Clear existing rows in the table
    print(status_table)
    print("Clearing table")
    # status_table.clear()
    # remove the current rows
    status_table.rows = []
    print(status_table)
    # Refresh local sequences
    downloader.load_local_sequences()

    # Add rows for each remote sequence
    for sequence in downloader.local_sequences:
        status = downloader.get_local_sequence_status(sequence)

        # Create status indicators
        status_icon = {
            "available": "✅ Complete",
            "incomplete": "⚠️ Incomplete",
            "not_found": "❌ Not Downloaded",
            "invalid": "❓ Invalid",
        }.get(status, "❓ Unknown")

        # Add the row to the table
        row = {"sequence": sequence, "status": status_icon}
        status_table.add_row(row)
        print(status_table)
    # Update the status label with the current directory information
    status_text = f"Current directory: {base_dir}\n"
    # status_text += f"Found {len(downloader.local_sequences)} local sequences"
    status_label.text = status_text


async def handle_directory_change(e):
    """Handle directory selection or input change."""

    new_dir = Path(dir_select.value)

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
                dir_select.value = str(downloader.base_dir)
                return
        else:
            # If user doesn't want to create it, revert to previous directory
            dir_select.value = str(downloader.base_dir)
            return

    # Update the downloader's base directory
    downloader.base_dir = new_dir  # Pass Path object directly instead of string
    downloader.load_local_sequences()  # Refresh local sequences list
    update_directory_status()


def create_dataset_manager_tab():
    """Create the Dataset Manager tab."""
    global status_label, dir_select, status_table

    # First, define all the functions we'll need
    async def download_sequence():
        if not sequence_select.value:
            ui.notify("Please select a sequence", type="warning")
            return

        # Check if directory exists before downloading
        if not Path(downloader.base_dir).exists():
            ui.notify("Selected directory does not exist. Please select a valid directory.", type="negative")
            return

        progress.visible = True
        status_label.text = f"Downloading sequence: {sequence_select.value}"

        # Use download_subfolder instead of download_sequence
        success = await asyncio.to_thread(downloader.download_subfolder, f"{sequence_select.value}/raw")

        if success:
            ui.notify("Sequence downloaded successfully!", type="positive")
            downloader.load_local_sequences()  # Refresh local sequences
            update_directory_status()  # Update status after successful download
        else:
            ui.notify("Failed to download sequence", type="negative")

        progress.visible = False

    async def download_ground_truth():
        if not site_select.value:
            ui.notify("Please select a site", type="warning")
            return

        # Check if directory exists before downloading
        if not Path(downloader.base_dir).exists():
            ui.notify("Selected directory does not exist. Please select a valid directory.", type="negative")
            return

        progress.visible = True
        status_label.text = f"Downloading ground truth for: {site_select.value}"

        success = await asyncio.to_thread(downloader.download_ground_truth, site_select.value)

        if success:
            ui.notify("Ground truth downloaded successfully!", type="positive")
            update_directory_status()  # Update status after successful download
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
            dir_select = ui.select(options=get_common_directories(), label="Select Directory", with_input=True)
            dir_select.props('style="width: 300px"')
            dir_select.value = str(downloader.base_dir)
            dir_select.on("update:model-value", handle_directory_change)

        # Status label
        status_label = ui.label("").classes("text-center q-mt-md q-mb-md whitespace-pre-line")

        # Create a table to display the status of sequences
        # status_table = ui.table(columns=["Sequence", "Status"], rows=[]).classes("q-mt-md")
        columns = [
            {"name": "sequence", "label": "Sequence", "field": "sequence", "required": True, "align": "left"},
            {"name": "status", "label": "Status", "field": "status", "sortable": True},
        ]
        rows = []
        status_table = ui.table(columns=columns, rows=rows, row_key="name")
        update_directory_status()

        # Sequence download section
        ui.label("Download Sequence").classes("text-h6 q-mb-md text-center")
        with ui.row().classes("justify-center"):
            sequence_select = ui.select(options=downloader.remote_sequences, label="Select Sequence", with_input=True)
            sequence_select.props('style="width: 300px"')

        # Add download sequence button here
        with ui.row().classes("justify-center q-mb-lg"):
            ui.button("Download Sequence", on_click=download_sequence).props("color=primary")

        # Ground truth download section
        ui.label("Download Ground Truth").classes("text-h6 q-mb-md text-center")
        with ui.row().classes("justify-center"):
            site_select = ui.select(options=downloader.list_available_sites(), label="Select Site", with_input=True)
            site_select.props('style="width: 300px"')

        # Progress bar
        progress = ui.linear_progress(0).classes("w-full")
        progress.visible = False

        # Only ground truth download button here
        with ui.row().classes("justify-center gap-4"):
            ui.button("Download Ground Truth", on_click=download_ground_truth).props("color=secondary")

    # Call the function to update the directory status
    update_directory_status()


def create_visualization_tab():
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


def create_analysis_tab():
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


def main():
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
            create_dataset_manager_tab()

        with ui.tab_panel("Visualization"):
            create_visualization_tab()

        with ui.tab_panel("Analysis"):
            create_analysis_tab()

    ui.run(host="0.0.0.0", port=8080)


if __name__ in {"__main__", "__mp_main__"}:
    main()
