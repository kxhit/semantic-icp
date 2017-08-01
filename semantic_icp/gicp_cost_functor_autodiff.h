#ifndef GICP_COST_FUNCTOR_AUTODIFF_HPP_
#define GICP_COST_FUNCTOR_AUTODIFF_HPP_

#include <iostream>
#include <algorithm>
#include <cmath>
#include <ceres/ceres.h>

namespace semanticicp {

    struct GICPCostFunctorAutoDiff {
        static const int K = 3;
        GICPCostFunctorAutoDiff ( const pcl::PointXYZ point_source,
                                  const pcl::PointXYZ point_target,
                                  const Eigen::Matrix3d cov_source,
                                  const Eigen::Matrix3d cov_target,
                                  const Sophus::SE3<double> base_transform) :
                                  point_source_ (point_source.x, point_source.y, point_source.z),
                                  point_target_ (point_target.x, point_target.y, point_target.z),
                                  cov_source_ (cov_source),
                                  cov_target_ (cov_target),
                                  base_transform_ (base_transform) {};

        template<class T>
        bool operator()(T const* const parameters,
                               T* residuals) const {
            Eigen::Map<Sophus::SE3<T> const> const transformIn_(parameters);
            Sophus::SE3<T> transform = transformIn_*base_transform_.cast<T>();
            Eigen::Matrix<T,3,3> R = transform.rotationMatrix();
            Eigen::Matrix<T,3,3> M = R*cov_source_.cast<T>();
            Eigen::Matrix<T,3,3> temp = M*R.transpose();
            temp += cov_target_.cast<T>();
            M = temp.inverse();

            Eigen::Matrix<T,3,1> transformed_point_source_ = transform*point_source_.cast<T>();
            Eigen::Matrix<T,3,1> res = transformed_point_source_-point_target_.cast<T>();
            Eigen::Matrix<T,3,1> dT = M*res;
            residuals[0] = T(res.transpose() * dT);
            return true;
        };

        Eigen::Vector3d point_source_;
        Eigen::Vector3d point_target_;
        Eigen::Matrix3d cov_source_;
        Eigen::Matrix3d cov_target_;
        Sophus::SE3<double> base_transform_;
    };

} // namspace semanticicp


#endif //GICP_COST_COST_FUNCTOR_HPP_
