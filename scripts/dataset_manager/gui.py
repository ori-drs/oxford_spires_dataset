from nicegui import ui
from nicegui.events import ValueChangeEventArguments

ui.label("OxSpires Dataset Manager")


def show(event: ValueChangeEventArguments):
    name = type(event.sender).__name__
    ui.notify(f"{name}: {event.value}")


ui.button("Button", on_click=lambda: ui.notify("Click"))
with ui.row():
    ui.checkbox("Checkbox", on_change=show)
    ui.switch("Switch", on_change=show)
ui.radio(["A", "B", "C"], value="A", on_change=show).props("inline")
with ui.row():
    ui.input("Text input", on_change=show)
    ui.select(["One", "Two"], value="One", on_change=show)
ui.link("And many more...", "/documentation").classes("mt-8")
# image_h = 300
# image_w = 500
# image_path = "/home/data/nerf_data_pipeline/2023-02-24-bodleian/raw/images/alphasense_driver_ros_cam0/1677224220.301245102.jpg"
# ui.image(image_path).props(f"width={image_w}px height={image_h}px")
ui.run(host="0.0.0.0", port=8080)
