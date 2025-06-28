#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <pcl/filters/filter_indices.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/cloud_iterator.h>

#include <Eigen/Dense>

namespace py = pybind11;


// PyBind
PYBIND11_MODULE(cpp, m) {
    m.def("add_test", [](int a, int b) {
        return a + b;
    }, "A function that adds two numbers");
}