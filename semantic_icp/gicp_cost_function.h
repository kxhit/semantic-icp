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
            Sophus::SE3d transform2_ = transform_*base_transform_;
            Eigen::Matrix3d R = transform2_.rotationMatrix();
            Eigen::Matrix3d M = R*cov_source_;
            Eigen::Matrix3d temp = M*R.transpose();
            temp += cov_target_;
            M = temp.inverse();

            Eigen::Vector3d transformed_point_source_ = transform2_*point_source_;
            Eigen::Vector3d res = transformed_point_source_-point_target_;
            Eigen::Vector3d dT = M*res;
            residuals[0] = double(res.transpose() * dT);

            if(jacobians!= NULL && jacobians[0] != NULL) {
                double *jacobian = jacobians[0];

                // dR is messy, can be cleaned up, taken from
                // http://www.matrixcalculus.org/
                Eigen::Matrix3d dR;
                dR = (M*res*point_source_.transpose()+cov_source_*R*M*res*(res.transpose()*M)
                     +cov_source_*R*M*res
                     *(res.transpose()*M)+M.transpose()*res*point_source_.transpose());

                dT *= 2.0;
                Eigen::Quaterniond dq = dRtodq(dR, transform2_.unit_quaternion());
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

        // Derivative of Rotation Matrix with respect to quaternion
        // Used in Chain Rule
        inline
        Eigen::Quaterniond dRtodq(const Eigen::Matrix3d dR, const Eigen::Quaterniond q)  const {
            Eigen::Quaterniond out;
            const double tx  = double(2)*q.x();
            const double ty  = double(2)*q.y();
            const double tz  = double(2)*q.z();
            const double tw  = double(2)*q.w();
            const double mfx  = double(-2)*tx;
            const double mfy  = double(-2)*ty;
            const double mfz  = double(-2)*tz;
            const double mtw  = double(-1)*tw;

            /* Eigen Quat to Rot
            res.coeffRef(0,0) = Scalar(1)-(tyy+tzz);
            res.coeffRef(0,1) = txy-twz;
            res.coeffRef(0,2) = txz+twy;
            res.coeffRef(1,0) = txy+twz;
            res.coeffRef(1,1) = Scalar(1)-(txx+tzz);
            res.coeffRef(1,2) = tyz-twx;
            res.coeffRef(2,0) = txz-twy;
            res.coeffRef(2,1) = tyz+twx;
            res.coeffRef(2,2) = Scalar(1)-(txx+tyy);
            */


            Eigen::Matrix3d dRdw;
            dRdw(0,0) = double(0);
            dRdw(0,1) = -tz;
            dRdw(0,2) = ty;
            dRdw(1,0) = tz;
            dRdw(1,1) = double(0);
            dRdw(1,2) = -tx;
            dRdw(2,0) = -ty;
            dRdw(2,1) = tx;
            dRdw(2,2) = double(0);

            out.w() = (dR.transpose()*dRdw).trace();

            Eigen::Matrix3d dRdx;
            dRdx(0,0) = double(0);
            dRdx(0,1) = ty;
            dRdx(0,2) = tz;
            dRdx(1,0) = ty;
            dRdx(1,1) = mfx;
            dRdx(1,2) = mtw;
            dRdx(2,0) = tz;
            dRdx(2,1) = tw;
            dRdx(2,2) = mfx;

            out.x() = (dR.transpose()*dRdx).trace();

            Eigen::Matrix3d dRdy;
            dRdy(0,0) = mfy;
            dRdy(0,1) = tx;
            dRdy(0,2) = tw;
            dRdy(1,0) = tx;
            dRdy(1,1) = double(0);
            dRdy(1,2) = tz;
            dRdy(2,0) = mtw;
            dRdy(2,1) = tz;
            dRdy(2,2) = mfy;

            out.y() = (dR.transpose()*dRdy).trace();

            Eigen::Matrix3d dRdz;
            dRdz(0,0) = mfz;
            dRdz(0,1) = mtw;
            dRdz(0,2) = tx;
            dRdz(1,0) = tw;
            dRdz(1,1) = mfz;
            dRdz(1,2) = ty;
            dRdz(2,0) = tx;
            dRdz(2,1) = ty;
            dRdz(2,2) = double(0);

            out.z() = (dR.transpose()*dRdz).trace();

            return out;
        }

    };

} // namspace semanticicp


#endif //GICP_COST_COST_FUNCTOR_HPP_
