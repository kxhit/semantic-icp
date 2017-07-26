#ifndef GICP_COST_FUNCTOR_HPP_
#define GICP_COST_FUNCTOR_HPP_

#include <iostream>
#include <algorithm>
#include <ceres/ceres.h>


namespace semanticicp {

    class GICPCostFunction :
    public ceres::SizedCostFunction<1,Sophus::SE3d::num_parameters>{
        public:
        static const int K = 3;
        GICPCostFunction ( const pcl::PointXYZ point_source,
                           const pcl::PointXYZ point_target,
                           const Eigen::Matrix3d cov_source,
                           const Eigen::Matrix3d cov_target,
                           const Sophus::SE3<double> base_transform) :
                           point_source_ (point_source.x, point_source.y, point_source.z),
                           point_target_ (point_target.x, point_target.y, point_target.z),
                           cov_source_ (cov_source),
                           cov_target_ (cov_target),
                           base_transform_ (base_transform) {};

        virtual bool Evaluate (double const* const* parameters,
                               double* residuals,
                               double** jacobians) const {
            Eigen::Map<Sophus::SE3<double> const> const transform_(parameters[0]);
            Eigen::Matrix3d R = transform_.rotationMatrix();
            Eigen::Matrix3d M = R*cov_source_;
            Eigen::Matrix3d temp = M*R.transpose();
            temp += cov_target_;
            M = temp.inverse();

            //std::cout << "Source: \n" << cov_source_ << std::endl;
            //std::cout << "Target: \n" << cov_target_ << std::endl;
            Eigen::Vector3d transformed_point_source_ = transform_*point_source_;
            Eigen::Vector3d res = transformed_point_source_-point_target_;
            Eigen::Vector3d dT = M*res;
            residuals[0] = double(res.transpose() * dT);
            //std::cout << "Mahal: \n" << M << std::endl;

            if(jacobians!= NULL && jacobians[0] != NULL) {
                double *jacobian = jacobians[0];
                Eigen::Vector3d base_point = base_transform_*point_source_;
                Eigen::Matrix3d dR = base_point * dT.transpose();
                dT *= 2.0;
                dR *= 2.0;
                Sophus::SE3<double> d_transform(dR, dT);
                std::copy(d_transform.data(),
                          d_transform.data() + Sophus::SE3d::num_parameters,
                          jacobian);
            }
            return true;
        };

        private:
        Eigen::Vector3d point_source_;
        Eigen::Vector3d point_target_;
        Eigen::Matrix3d cov_source_;
        Eigen::Matrix3d cov_target_;
        Sophus::SE3<double> base_transform_;
    };

} // namspace semanticicp


#endif //GICP_COST_COST_FUNCTOR_HPP_
