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
                //Eigen::Vector3d base_point = base_transform_*point_source_;
                Eigen::Matrix3d dR = Eigen::Matrix3d::Zero();
                Eigen::Matrix3d J;
                Eigen::Matrix3d dM;
                Eigen::Vector3d dres;
                for(int i = 0; i < 3; i++) {
                    for(int j = 0; j < 3; j++) {
                        J.setZero();
                        J(i,j) = 1;
                        dM = R.transpose()*cov_source_*J+J.transpose()*cov_source_*R;
                        dres =  J*res;
                        double v = res.dot(M*dres)+res.dot(M*dM*M*res);
                        dR(i,j) = v;
                    }
                }
                dT *= 2.0;
                dR *= 2.0;
                Eigen::Quaterniond dq = dRtodq(dR, transform_.unit_quaternion());
                jacobian[4] = dT(0);
                jacobian[5] = dT(1);
                jacobian[6] = dT(2);
                jacobian[3] = dq.w();
                jacobian[0] = dq.x();
                jacobian[1] = dq.y();
                jacobian[2] = dq.z();
            }
            return true;
        };

        private:
        Eigen::Vector3d point_source_;
        Eigen::Vector3d point_target_;
        Eigen::Matrix3d cov_source_;
        Eigen::Matrix3d cov_target_;
        Sophus::SE3<double> base_transform_;

        Eigen::Quaterniond dRtodq(const Eigen::Matrix3d dR, const Eigen::Quaterniond q)  const {
            Eigen::Quaterniond out;
            double w = q.w();
            double x = q.x();
            double y = q.y();
            double z = q.z();

            Eigen::Matrix3d dRdw;
            dRdw <<     0.0, -2.0*z,  2.0*y,
                      2.0*z,    0.0, -2.0*x,
                     -2.0*y,  2.0*x,    0.0;
            out.w() = (dR.array()*dRdw.array()).sum();

            Eigen::Matrix3d dRdx;
            dRdx <<     0.0,  2.0*y,  2.0*z,
                      2.0*y, -4.0*x, -2.0*w,
                      2.0*z,  2.0*w, -4.0*x;
            out.x() = (dR.array()*dRdx.array()).sum();

            Eigen::Matrix3d dRdy;
            dRdy <<  -4.0*y,  2.0*x,  2.0*w,
                      2.0*x,    0.0,  2.0*z,
                     -2.0*w,  2.0*z, -4.0*y;
            out.y() = (dR.array()*dRdy.array()).sum();

            Eigen::Matrix3d dRdz;
            dRdz <<  -4.0*z, -2.0*w,  2.0*x,
                      2.0*w, -4.0*z,  2.0*y,
                      2.0*x,  2.0*y,    0.0;
            out.z() = (dR.array()*dRdz.array()).sum();

            return out;
        }

    };

} // namspace semanticicp


#endif //GICP_COST_COST_FUNCTOR_HPP_
