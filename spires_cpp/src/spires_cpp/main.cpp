#include "spires_cpp/octomap/filter_pcd_with_bt.h"
#include "spires_cpp/octomap/get_occ_free_from_bt.h"
#include "spires_cpp/octomap/pcd2bt.h"
#include "spires_cpp/octomap/progress_bar.h"
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pybind11/pybind11.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

int add(int i, int j) { return i + j; }

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
  m.def("processPCDFolder", &processPCDFolder, "Process PCD files and save to OctoMap", py::arg("folderPath"),
        py::arg("resolution"), py::arg("save_path"));
  m.def("removeUnknownPoints", &removeUnknownPoints, "Remove unknown points from PCD file", py::arg("input_pcd_path"),
        py::arg("input_bt_path"), py::arg("output_file_path"));
  m.def("convertOctreeToPointCloud", &convertOctreeToPointCloud, "Convert Octree to PointCloud",
        py::arg("bt_file_path"), py::arg("free_pcd_path"), py::arg("occupied_pcd_path"));
  py::class_<octomap::OcTree>(m, "OcTree")
      .def(py::init<double>(), py::arg("resolution"))
      .def(py::init<std::string>(), py::arg("filename"))
      .def("readBinary", static_cast<bool (octomap::OcTree::*)(const std::string &)>(&octomap::OcTree::readBinary))
      .def("writeBinary", static_cast<bool (octomap::OcTree::*)(const std::string &)>(&octomap::OcTree::writeBinary))
      .def("getResolution", &octomap::OcTree::getResolution)
      .def("size", &octomap::OcTree::size)
      .def("getTreeDepth", &octomap::OcTree::getTreeDepth);

#ifdef VERSION_INFO
  m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
  m.attr("__version__") = "dev";
#endif
}
