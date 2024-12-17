#include <mutex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

class RingBuffer {
public:
    RingBuffer(int cap = 1024): cap(cap), end(0), buf(cap) {
        id_start["default"] = 0;
    }

    void reset() {
        std::lock_guard<std::mutex> lock(mutex);
        end = 0;
        id_start.clear();
        id_start["default"] = 0;
    }

    void push(py::array_t<double> item) {
        std::lock_guard<std::mutex> lock(mutex);
        buf[end % cap] = item;
        end++;
    }

    py::array_t<double> peek(int index) {
        if (index >= end || index < end - cap) {
            return py::array_t<double>(); // Return an empty array
        }
        int place = index % cap;
        return buf[place];
    }

    std::vector<py::array_t<double>> pull(const std::string& client_id = "default") {
        std::lock_guard<std::mutex> lock(mutex);
        if (id_start.find(client_id) == id_start.end()) {
            id_start[client_id] = 0;
        }

        std::vector<py::array_t<double>> ret;
        if (id_start[client_id] >= end) {
            return ret;
        }
        if (id_start[client_id] < end - cap) {
            id_start[client_id] = end - cap;
        }
        for (int idx = id_start[client_id]; idx < end; ++idx) {
            ret.push_back(peek(idx));
        }
        id_start[client_id] = end;
        return ret;
    }

    py::array_t<double> pop_front(const std::string& client_id = "default") {
        std::lock_guard<std::mutex> lock(mutex);
        if (id_start.find(client_id) == id_start.end()) {
            id_start[client_id] = 0;
        }

        if (id_start[client_id] >= end) {
            return py::array_t<double>(); // Return an empty array
        }
        if (id_start[client_id] < end - cap) {
            id_start[client_id] = end - cap;
        }
        return peek(id_start[client_id]++);
    }

    int get_valid_len(const std::string& client_id = "default") {
        std::lock_guard<std::mutex> lock(mutex);
        if (id_start.find(client_id) == id_start.end()) {
            id_start[client_id] = 0;
        }
        return end - std::max(end - cap, id_start[client_id]);
    }

    int get_cap() const {
        return cap;
    }

    int get_end() const {
        return end;
    }

private:
    int cap;
    int end;
    std::vector<py::array_t<double>> buf;
    std::unordered_map<std::string, int> id_start;
    std::mutex mutex;
};

PYBIND11_MODULE(ring_buffer, m) {
    py::class_<RingBuffer>(m, "RingBuffer")
        .def(py::init<int>(), py::arg("cap") = 1024)
        .def("reset", &RingBuffer::reset)
        .def("push", &RingBuffer::push, py::arg("item"))
        .def("peek", &RingBuffer::peek, py::arg("index"))
        .def("pop_front", &RingBuffer::pop_front, py::arg("client_id") = "default")
        .def("pull", &RingBuffer::pull, py::arg("client_id") = "default")
        .def("get_valid_len", &RingBuffer::get_valid_len, py::arg("client_id") = "default")
        .def("get_cap", &RingBuffer::get_cap)
        .def("get_end", &RingBuffer::get_end);
}
