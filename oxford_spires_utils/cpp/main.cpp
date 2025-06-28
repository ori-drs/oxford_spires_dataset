#include <vector>

#include <pybind11/pybind11.h>

#include <octomap/OcTree.h>
#include <octomap/octomap.h>

namespace py = pybind11;


// PyBind
PYBIND11_MODULE(cpp, m) {
    m.def("add_test", [](int a, int b) {
        return a + b;
    }, "A function that adds two numbers");
    py::class_<octomap::OcTree>(m, "OcTree")
    .def(py::init<double>(), py::arg("resolution"))
    .def(py::init<std::string>(), py::arg("filename"))
    .def("readBinary", static_cast<bool (octomap::OcTree::*)(const std::string &)>(&octomap::OcTree::readBinary))
    .def("writeBinary", static_cast<bool (octomap::OcTree::*)(const std::string &)>(&octomap::OcTree::writeBinary))
    .def("getResolution", &octomap::OcTree::getResolution)
    .def("size", &octomap::OcTree::size)
    .def("getTreeDepth", &octomap::OcTree::getTreeDepth);
}