#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

using std::vector;
typedef double Flt;

namespace itp_state {

using std::cout;
using std::endl;

Flt slow_pow(Flt x, int y) {
    Flt result = 1.0;
    for (int i = 0; i < y; ++i) {
        result *= x;
    }
    return result;
}

Flt limited_by(Flt x, Flt x_abs_max) {
    return std::clamp(x, -x_abs_max, x_abs_max);
}

int sign_func(Flt x) {
    return (x >= 0) ? 1 : -1;
}

Flt poly_func(Flt t, const vector<Flt>& coe) {
    Flt result = 0.0;
    for (size_t i = 0; i < coe.size(); ++i) {
        result += coe[i] * std::pow(t, static_cast<Flt>(i));
    }
    return result;
}

// 多项式求导函数
vector<Flt> poly_derivative(const vector<Flt>& coe) {
    vector<Flt> deriv;
    if (coe.size() <= 1) {
        deriv.push_back(0.0);
        return deriv;
    }
    for (size_t i = 1; i < coe.size(); ++i) {
        deriv.push_back(coe[i] * i);
    }
    return deriv;
}

Flt newton_raphson(
    std::function<Flt(Flt)> func,
    std::function<Flt(Flt)> func_deriv,
    Flt initial,
    int max_iter = 50,
    Flt tol = 1e-6
) {
    Flt x = initial;
    for (int i = 0; i < max_iter; ++i) {
        Flt fx = func(x);
        Flt dfx = func_deriv(x);
        if (std::abs(dfx) < tol) {
            break;
        }
        Flt x_new = x - fx / dfx;
        if (std::abs(x_new - x) < tol) {
            break;
        }
        x = x_new;
    }
    return x;
}

std::pair<Flt, Flt> sug_inv_t(Flt d, Flt v0, Flt a0, Flt v_m1, Flt a_m, Flt j_m) {
    if (a_m * a_m >= v_m1 * j_m) {
        Flt d1 = j_m * std::pow(v_m1 / j_m, 1.5f) / 3.0;
        Flt d2 = v_m1 * std::sqrt(v_m1 / j_m);
        if (d >= d2) {
            return { v_m1, 0.0 };
        }
        if (d >= d1) {
            vector<Flt> coe = { j_m * std::pow(v_m1 / j_m, 1.5f) / 3.0,
                                -v_m1,
                                j_m * std::sqrt(v_m1 / j_m),
                                -j_m / 6.0 };

            // 定义函数和导数
            auto func = [&](Flt t) -> Flt { return poly_func(t, coe) - d; };
            // 计算导数系数
            vector<Flt> deriv_coe = poly_derivative(coe);
            auto func_deriv = [&](Flt t) -> Flt { return poly_func(t, deriv_coe); };

            Flt t_sug = newton_raphson(func, func_deriv, a_m / j_m);
            Flt v_sug = -j_m * t_sug * t_sug / 2.0 + 2.0 * j_m * t_sug * std::sqrt(v_m1 / j_m)
                - 2.0 * j_m * v_m1 / j_m + v_m1;
            Flt a_sug = j_m * std::sqrt(v_m1 / j_m) - j_m * (t_sug - std::sqrt(v_m1 / j_m));
            return { v_sug, a_sug };
        }
        Flt t_sug = std::pow(6.0, 1.0 / 3.0) * std::pow(d / j_m, 1.0 / 3.0);
        Flt v_sug = j_m * t_sug * t_sug / 2.0;
        Flt a_sug = j_m * t_sug;
        return { v_sug, a_sug };
    } else {
        Flt t1 = a_m / j_m;
        Flt t2 = v_m1 / a_m;
        Flt t3 = a_m / j_m + v_m1 / a_m;
        Flt d1 = std::pow(a_m, 3) / (6.0 * j_m * j_m);
        Flt d2 = std::pow(a_m, 3) / (6.0 * j_m * j_m) - a_m * v_m1 / (2.0 * j_m)
            + (v_m1 * v_m1) / (2.0 * a_m);
        Flt d3 = v_m1 * (a_m * a_m + j_m * v_m1) / (2.0 * a_m * j_m);
        if (d >= d3) {
            return { v_m1, 0.0 };
        }
        if (d >= d2) {
            vector<Flt> coe = { (std::pow(a_m, 6) + std::pow(j_m, 3) * std::pow(v_m1, 3))
                                    / (6.0 * std::pow(a_m, 3) * j_m * j_m),
                                (-(std::pow(a_m, 4))-j_m * j_m * v_m1 * v_m1)
                                    / (2.0 * a_m * a_m * j_m),
                                (a_m * a_m + j_m * v_m1) / (2.0 * a_m),
                                -j_m / 6.0 };

            // 定义函数和导数
            auto func = [&](Flt t) -> Flt { return poly_func(t, coe) - d; };
            // 计算导数系数
            vector<Flt> deriv_coe = poly_derivative(coe);
            auto func_deriv = [&](Flt t) -> Flt { return poly_func(t, deriv_coe); };

            Flt t_sug = newton_raphson(func, func_deriv, t2);
            Flt v_sug = -(a_m * a_m) / (2.0 * j_m) + a_m * t_sug - j_m * t_sug * t_sug / 2.0
                + j_m * t_sug * v_m1 / a_m - j_m * v_m1 * v_m1 / (2.0 * a_m * a_m);
            Flt a_sug = a_m - j_m * (t_sug - v_m1 / a_m);
            return { v_sug, a_sug };
        }
        if (d >= d1) {
            Flt discriminant = 3.0 * a_m * a_m
                + std::sqrt(3.0) * std::sqrt(a_m * (-std::pow(a_m, 3) + 24.0 * d * j_m * j_m));
            Flt t_sug = discriminant / (6.0 * a_m * j_m);
            Flt v_sug = -(a_m * a_m) / (2.0 * j_m) + a_m * t_sug;
            return { v_sug, a_m };
        }
        Flt t_sug = std::pow(6.0, 1.0 / 3.0) * std::pow(d / j_m, 1.0 / 3.0);
        Flt v_sug = j_m * t_sug * t_sug / 2.0;
        Flt a_sug = j_m * t_sug;
        return { v_sug, a_sug };
    }
}

Flt itpltn_best_v_a(
    Flt x0,
    Flt v0,
    Flt a0,
    Flt x_tar,
    Flt v_tar,
    Flt f,
    Flt v_abs_max,
    Flt a_abs_max,
    Flt unknown
) {
    // 不修改原始数据
    Flt d_h = x_tar - (x0 + x0 + v0 / f + 0.5f * a0 / (f * f)) / 2.0;
    int dir = (d_h < 0) ? -1 : 1;
    d_h *= dir;
    v0 *= dir;
    a0 *= dir;
    v_tar *= dir;

    v0 -= v_tar;
    Flt v_max = std::max(v_abs_max - v_tar, 0.0);
    Flt v_min = std::min(-v_abs_max - v_tar, 0.0);

    Flt v_sug = limited_by(std::sqrt(2.0 * a_abs_max * d_h), v_max);
    Flt a_limited = limited_by((v_sug - v0) * f, a_abs_max * unknown);
    return a_limited * dir;
}

Flt itpltn_best_v_j(
    Flt x0,
    Flt v0,
    Flt a0,
    Flt x_tar,
    Flt v_tar,
    Flt f,
    Flt v_abs_max,
    Flt a_abs_max,
    Flt j_abs_max
) {
    Flt d_h = x_tar - (x0 + x0 + v0 / f + 0.5f * a0 / (f * f)) / 2.0;
    int dir = (d_h < 0) ? -1 : 1;
    d_h *= dir;
    v0 *= dir;
    a0 *= dir;
    v_tar *= dir;

    v0 -= v_tar;
    Flt v_max = std::max(v_abs_max - v_tar, 0.0);
    Flt v_min = std::min(-v_abs_max - v_tar, 0.0);

    auto [v_sug, a_sug] = sug_inv_t(d_h, v0, a0, v_max, a_abs_max, j_abs_max);
    a_sug = -a_sug;

    Flt j = itpltn_best_v_a(
        v0,
        a0,
        0.0, // np.zeros_like(x0) 替换为单一值
        v_sug,
        // a_sug * np.clip(v0 / v_sug, xxx, 1), this is very tricky
        a_sug * std::clamp(v0 / v_sug, 2.0 / 3.0, 1.0),
        f,
        a_abs_max,
        j_abs_max,
        1.1f
    );
    return j * dir;
}

class ItpState {
public:
    vector<Flt> v_max;
    vector<Flt> a_max;
    vector<Flt> j_max;
    Flt fps = 0.0;
    vector<Flt> pre_sent_x;
    vector<Flt> pre_sent_v;
    vector<Flt> pre_sent_a;
    vector<Flt> pre_sent_j;

    ItpState() = default;

    void init(
        const vector<Flt>& x0 = {},
        const vector<Flt>& v0 = {},
        const vector<Flt>& v_max = {},
        const vector<Flt>& a_max = {},
        const vector<Flt>& j_max = {},
        Flt fps = 0.0
    ) {
        if (!x0.empty()) {
            this->pre_sent_x = x0;
        }
        if (!v0.empty()) {
            this->pre_sent_v = v0;
        }
        if (!v_max.empty()) {
            this->v_max = v_max;
        }
        if (!a_max.empty()) {
            this->a_max = a_max;
        }
        if (!j_max.empty()) {
            this->j_max = j_max;
        }
        if (fps != 0.0) {
            this->fps = fps;
        }

        if (!x0.empty() || !v0.empty()) {
            size_t size = !x0.empty() ? x0.size() : v0.size();
            this->pre_sent_a = vector<Flt>(size, 0.0);
            this->pre_sent_j = vector<Flt>(size, 0.0);
        }
    }

    // 返回类型为 vector of tuples: (so_x, so_v, so_a, j)
    vector<std::tuple<vector<Flt>, vector<Flt>, vector<Flt>, vector<Flt>>> interpolate(
        const vector<Flt>& x_tar,
        const vector<Flt>& v_tar,
        int points_needed,
        Flt first_delta_t
    ) {
        if (this->v_max.empty() || this->a_max.empty() || this->j_max.empty() || this->fps == 0.0
            || this->pre_sent_x.empty() || this->pre_sent_v.empty() || this->pre_sent_a.empty()
            || this->pre_sent_j.empty())
        {
            throw std::invalid_argument("ItpState not initialized");
        }

        if (x_tar.size() != v_tar.size()) {
            throw std::invalid_argument("x_tar and v_tar must have the same size");
        }

        size_t dims = x_tar.size();
        vector<std::tuple<vector<Flt>, vector<Flt>, vector<Flt>, vector<Flt>>> ret;
        ret.reserve(points_needed);

        // 初始化结果存储
        vector<vector<Flt>> so_x_points(points_needed, vector<Flt>(dims, 0.0));
        vector<vector<Flt>> so_v_points(points_needed, vector<Flt>(dims, 0.0));
        vector<vector<Flt>> so_a_points(points_needed, vector<Flt>(dims, 0.0));
        vector<vector<Flt>> j_points(points_needed, vector<Flt>(dims, 0.0));

        for (size_t dim = 0; dim < dims; ++dim) {
            for (int i = 0; i < points_needed; ++i) {
                Flt current_delta_t = first_delta_t + static_cast<Flt>(i) / this->fps;
                Flt t_tar = current_delta_t;
                Flt x_tar_t = x_tar[dim] + v_tar[dim] * t_tar;
                Flt v_tar_t = v_tar[dim];

                Flt j = itpltn_best_v_j(
                    this->pre_sent_x[dim],
                    this->pre_sent_v[dim],
                    this->pre_sent_a[dim],
                    x_tar_t,
                    v_tar_t,
                    this->fps,
                    this->v_max[dim],
                    this->a_max[dim],
                    this->j_max[dim]
                );

                Flt so_a = this->pre_sent_a[dim] + j / this->fps;
                Flt so_v = this->pre_sent_v[dim] + so_a / this->fps;
                Flt so_x = this->pre_sent_x[dim] + so_v / this->fps;

                j_points[i][dim] = j;
                so_a_points[i][dim] = so_a;
                so_v_points[i][dim] = so_v;
                so_x_points[i][dim] = so_x;

                // 更新前一次发送的状态
                this->pre_sent_j[dim] = j;
                this->pre_sent_a[dim] = so_a;
                this->pre_sent_v[dim] = so_v;
                this->pre_sent_x[dim] = so_x;
            }
        }

        // 组合每个点的结果
        for (int i = 0; i < points_needed; ++i) {
            ret.emplace_back(so_x_points[i], so_v_points[i], so_a_points[i], j_points[i]);
        }

        return ret;
    }
};
} // namespace itp_state

namespace py = pybind11;

PYBIND11_MODULE(itp_state, m) {
    m.doc() = "ITP State module";

    // 绑定 ItpState 类
    py::class_<itp_state::ItpState>(m, "ItpState")
        .def(py::init<>())
        .def(
            "init",
            &itp_state::ItpState::init,
            py::arg("x0") = std::vector<Flt>(),
            py::arg("v0") = std::vector<Flt>(),
            py::arg("v_max") = std::vector<Flt>(),
            py::arg("a_max") = std::vector<Flt>(),
            py::arg("j_max") = std::vector<Flt>(),
            py::arg("fps") = 0.0
        )
        .def(
            "interpolate",
            &itp_state::ItpState::interpolate,
            py::arg("x_tar"),
            py::arg("v_tar"),
            py::arg("points_needed"),
            py::arg("first_delta_t") = 0.0
        );
}
