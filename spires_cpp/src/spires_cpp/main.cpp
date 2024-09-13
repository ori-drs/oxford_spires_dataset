#include <pybind11/pybind11.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include "progress_bar.h"
#include "pcd2bt.h"

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) {
    return i + j;
}

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        Pybind11 example plugin
        -----------------------

        .. currentmodule:: scikit_build_example

        .. autosummary::
           :toctree: _generate

           add
           subtract
    )pbdoc";
    m.def("processPCDFolder", &processPCDFolder, "Process PCD files and save to OctoMap",
        py::arg("folderPath"), py::arg("resolution"), py::arg("save_path"));
    // m.def("processPCDFolder", &processPCDFolder, "Process PCD files and save to OctoMap",
    //     py::arg("folderPath"), py::arg("resolution"), py::arg("save_path"));
#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}