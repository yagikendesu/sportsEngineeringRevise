////
//// Created by yagi on 19/02/21.
////
//#include "panorama.h"
//#include <ceres/ceres.h>
//
//using namespace std;
//using namespace yagi;
//using ceres::AutoDiffCostFunction;
//using ceres::CostFunction;
//using ceres::Problem;
//using ceres::Solver;
//using namespace ceres;
//
//double a = 0.0;
//double b = 0.0;
//double c = 0.0;
//double d = 0.0;
//
//template<typename TYPE, std::size_t SIZE>
//std::size_t array_length(const TYPE (&array)[SIZE])
//{
//    return SIZE;
//}
//
//struct ExponentialResidual {
//    ExponentialResidual(double x, double y)
//            : x_(x), y_(y) {}
//
//    template <typename T>
//    bool operator()(const T* const a,
//                    const T* const b,
//                    const T* const c,
//                    const T* const d,
//                    T* residual) const {
////        residual[0] = T(y_) - exp(m[0] * T(x_) + c[0]);
//        residual[0] = T(y_) - a[0]*(exp((T(x_) + b[0])/c[0]) - exp(-T(x_)/d[0]));
//        return true;
//    }
//
//    private:
//        // Observations for a sample.
//        const double x_;
//        const double y_;
//};
//
//void Panorama::curveFitting(vector<double> speed10mList){
//
//    double a = 0.0;
//    double b = 0.0;
//    double c = 0.0;
//    double d = 0.0;
//
//    Problem problem;
//    for (int i = 0; i < speed10mList.size(); ++i) {
//        double dist = i*10;
//        CostFunction* cost_function =
//                new AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(new ExponentialResidual(dist, speed10mList[i]));
//        //                        new ExponentialResidual(data[2 * i], data[2 * i + 1]));
//
//        problem.AddResidualBlock(cost_function, NULL, &a, &b, &c, &d);
//
//        //制約設定
//        problem.SetParameterLowerBound(&a,0,12.0);
//        problem.SetParameterUpperBound(&b,0,14.0);
//    }
//
//    //ソルバの準備
//    Solver::Options options;
//
//    //CXSparseをインストールしていないときはDENSE_*にしないとエラーになる。
//    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
//
//    //最適化
//    Solver::Summary summary;
//    ceres::Solve(options, &problem, &summary);
//
//    cout << "Fit complete:   a: " << a
//         << "Fit complete:   b: " << b
//         << "Fit complete:   c: " << c
//         << "Fit complete:   d: " << d <<"\n" << endl;
//
//
//}