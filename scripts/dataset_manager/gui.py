import asyncio
import os
from pathlib import Path

from dataset_downloader import DatasetDownloader
from nicegui import ui


def get_common_directories():
    """Get a list of common directories to choose from."""
    common_dirs = [
        str(Path.home()),
        str(Path.home() / "data"),
        str(Path.home() / "workspace"),
        str(Path.home() / "Downloads"),
        "/data",
        "/home/data",
    ]
    return [d for d in common_dirs if os.path.exists(d)]


# Initialize the dataset downloader
downloader = DatasetDownloader()


def update_directory_status():
    """Update the status information for the current directory."""
    base_dir = Path(downloader.base_dir)
    if not base_dir.exists():
        status_label.text = f"Directory does not exist: {base_dir}"
        return

    # Count files and directories
    files = list(base_dir.rglob("*"))
    dirs = [f for f in files if f.is_dir()]
    files = [f for f in files if f.is_file()]

    # Check if directory is empty
    is_empty = len(files) == 0 and len(dirs) == 0

    status_text = f"Current directory: {base_dir}\n"
    status_text += f"Status: {'Empty' if is_empty else 'Not Empty'}\n"
    status_text += f"Number of files: {len(files)}\n"
    status_text += f"Number of directories: {len(dirs)}"

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
    update_directory_status()


# Create a column with max width and center alignment
with ui.column().classes("max-w-2xl mx-auto p-4"):
    ui.label("OxSpires Dataset Manager").classes("text-h4 q-mb-md text-center")

    # Directory selection section
    ui.label("Select Base Directory").classes("text-h6 q-mb-md text-center")
    with ui.row().classes("justify-center"):
        dir_select = ui.select(options=get_common_directories(), label="Select Directory", with_input=True)
        dir_select.props('style="width: 300px"')

        # Set initial value
        dir_select.value = str(downloader.base_dir)

        # Handle directory changes
        dir_select.on("update:model-value", handle_directory_change)

    # Directory status
    status_label = ui.label("").classes("text-center q-mt-md q-mb-md whitespace-pre-line")

    # Sequence download section
    ui.label("Download Sequence").classes("text-h6 q-mb-md text-center")
    with ui.row().classes("justify-center"):
        sequence_select = ui.select(
            options=downloader.list_available_sequences(), label="Select Sequence", with_input=True
        )
        sequence_select.props('style="width: 300px"')

    # Ground truth download section
    ui.label("Download Ground Truth").classes("text-h6 q-mb-md text-center")
    with ui.row().classes("justify-center"):
        site_select = ui.select(options=downloader.list_available_sites(), label="Select Site", with_input=True)
        site_select.props('style="width: 300px"')

    # Progress bar
    progress = ui.linear_progress(0).classes("w-full")
    progress.visible = False

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

        success = await asyncio.to_thread(downloader.download_sequence, sequence_select.value)

        if success:
            ui.notify("Sequence downloaded successfully!", type="positive")
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

    # Download buttons
    with ui.row().classes("justify-center gap-4"):
        ui.button("Download Sequence", on_click=download_sequence).props("color=primary")
        ui.button("Download Ground Truth", on_click=download_ground_truth).props("color=secondary")

    # Initialize directory status
    update_directory_status()

ui.run(host="0.0.0.0", port=8080)
