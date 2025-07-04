#include "oxspires_tools/filter_pcd_with_bt.h"
#include "oxspires_tools/get_occ_free_from_bt.h"
#include "oxspires_tools/pcd2bt.h"
#include "oxspires_tools/progress_bar.h"

#include <pybind11/pybind11.h>


namespace py = pybind11;


// PyBind
PYBIND11_MODULE(cpp, m) {
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
}