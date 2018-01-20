#ifndef GICP_HPP_
#define GICP_HPP_

#include <iostream>

#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>

//#include <gicp_cost_functor_autodiff.h>
#include <gicp_cost_function.h>
#include <local_parameterization_se3.h>

#include<Eigen/StdVector>
#include<ceres/gradient_checker.h>

namespace semanticicp
{

template <typename PointT>
void GICP<PointT>::align(PointCloudPtr finalCloud) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    Sophus::SE3d init(mat);
    align(finalCloud, init);
};

template <typename PointT>
void GICP<PointT>::align(
        PointCloudPtr finalCloud, Sophus::SE3d &initTransform) {

    computeCovariances(sourceCloud_, sourceKdTree_, sourceCovariances_);
    computeCovariances(targetCloud_, targetKdTree_, targetCovariances_);

    Sophus::SE3d currentTransform(initTransform);
    bool converged = false;
    size_t count = 0;

    while(converged!=true) {
        std::vector<Sophus::SE3d> transformsVec;
        CovarianceVector covVec;

        // Build The Problem
        ceres::Problem problem;

        // Add Sophus SE3 Parameter block with local parametrization
        Sophus::SE3d estTransform(currentTransform);
        problem.AddParameterBlock(estTransform.data(), Sophus::SE3d::num_parameters,
                                  new LocalParameterizationSE3);

        double mseHigh = 0;

        typename pcl::PointCloud<PointT>::Ptr transformedSource (new pcl::PointCloud<PointT>());
        Sophus::SE3d transform = currentTransform;
        Eigen::Matrix4d transMat = transform.matrix();
        pcl::transformPointCloud(*sourceCloud_,
                                    *transformedSource,
                                    transMat);

        std::vector<int> targetIndx;
        std::vector<float> distSq;


        std::cout << "Num Points: " << transformedSource->size() << std::endl;
        for(int sourceIndx = 0; sourceIndx != transformedSource->size(); sourceIndx++) {
            const PointT &transformedSourcePoint = transformedSource->points[sourceIndx];

            targetKdTree_->nearestKSearch(transformedSourcePoint, 1, targetIndx, distSq);
                if( distSq[0] < 250 ) {
                    const PointT &sourcePoint =
                        sourceCloud_->points[sourceIndx];
                    const Eigen::Matrix3d &sourceCov =
                        sourceCovariances_->at(sourceIndx);
                    const PointT &targetPoint =
                        targetCloud_->points[targetIndx[0]];
                    const Eigen::Matrix3d &targetCov =
                        targetCovariances_->at(targetIndx[0]);

                    //   Autodif Cost function
                    //GICPCostFunctorAutoDiff *c= new GICPCostFunctorAutoDiff(sourcePoint,
                    //                                                       targetPoint,
                    //                                                       sourceCov,
                    //                                                       targetCov,
                    //                                                       baseTransformation_);
                    //ceres::CostFunction* cost_function =
                    //    new ceres::AutoDiffCostFunction<GICPCostFunctorAutoDiff,
                    //                                    1,
                    //                                    Sophus::SE3d::num_parameters>(c);

                    //   Analytical Cost Function
                    GICPCostFunction* cost_function = new GICPCostFunction(sourcePoint,
                                                                           targetPoint,
                                                                           sourceCov,
                                                                           targetCov,
                                                                           baseTransformation_);

                    problem.AddResidualBlock(cost_function,
                                             new ceres::CauchyLoss(1.5),
                                             estTransform.data());
                    // Gradient Check
                    if (false) {
                        ceres::NumericDiffOptions numeric_diff_options;
                        numeric_diff_options.relative_step_size = 1e-13;

                        std::vector<const ceres::LocalParameterization*> lp;
                        lp.push_back(new LocalParameterizationSE3);

                        ceres::GradientChecker gradient_checker(cost_function,
                                    &lp,
                                    numeric_diff_options);

                        ceres::GradientChecker::ProbeResults results;
                        std::vector<double *> params;
                        params.push_back(estTransform.data());
                        if (!gradient_checker.Probe(params.data(), 5e-4, &results)) {
                                std::cout << "An error has occurred:\n";
                                std::cout << results.error_log;
                                std::cout << results.jacobians[0] << std::endl;
                                std::cout << results.numeric_jacobians[0] << std::endl;
                                std::cout << estTransform.matrix() << std::endl;
                                std::cout << sourcePoint << std::endl;
                                std::cout << targetPoint << std::endl;
                                std::cout << sourceCov << std::endl;
                                std::cout << targetCov << std::endl;

                        }
                    }

                }

        } // For loop over points
        // Sovler Options
        ceres::Solver::Options options;
        options.gradient_tolerance = 0.1 * Sophus::Constants<double>::epsilon();
        options.function_tolerance = 0.1 * Sophus::Constants<double>::epsilon();
        options.linear_solver_type = ceres::DENSE_QR;
        options.num_threads = 4;
        options.max_num_iterations = 400;
       // options.check_gradients = true;
        options.gradient_check_numeric_derivative_relative_step_size = 1e-8;
        options.gradient_check_relative_precision = 1e-6;

        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        std::cout << summary.FullReport() << std::endl;

        double mse = (currentTransform.inverse()*estTransform).log().squaredNorm();
        if(mse < 1e-3 || count>50)
            converged = true;
        std::cout<< "MSE: " << mse << std::endl;
        std::cout<< "Transform: " << std::endl;
        std::cout<< estTransform.matrix() << std::endl;
        std::cout<< "Itteration: " << count << std::endl;
        currentTransform = estTransform;
        count++;
    }

    finalTransformation_ = currentTransform;

    Sophus::SE3d trans = finalTransformation_*baseTransformation_;
    Eigen::Matrix4f mat = (trans.matrix()).cast<float>();
    if( finalCloud != nullptr ) {
        pcl::transformPointCloud(*sourceCloud_,
                                 *finalCloud,
                                 mat);
    }

    outer_iter = count;
};

template <typename PointT>
void GICP<PointT>::computeCovariances(const PointCloudPtr cloudptr,
                                      KdTreePtr treeptr, MatricesVectorPtr matvecptr) {

    // Variables for computing Covariances
    Eigen::Vector3d mean;
    std::vector<int> nn_idecies; nn_idecies.reserve (kCorrespondences_);
    std::vector<float> nn_dist_sq; nn_dist_sq.reserve (kCorrespondences_);

    // Set up Itteration
    matvecptr->resize(cloudptr->size());

    for(size_t itter = 0; itter < cloudptr->size(); itter++) {
        const PointT &query_pt = (*cloudptr)[itter];

        Eigen::Matrix3d cov;
        cov.setZero();
        mean.setZero();

        treeptr->nearestKSearch(query_pt, kCorrespondences_, nn_idecies, nn_dist_sq);

        for( int index: nn_idecies) {
            const PointT &pt = (*cloudptr)[index];

            mean[0] += pt.x;
            mean[1] += pt.y;
            mean[2] += pt.z;

            cov(0,0) += pt.x*pt.x;

            cov(1,0) += pt.y*pt.x;
            cov(1,1) += pt.y*pt.y;

            cov(2,0) += pt.z*pt.x;
            cov(2,1) += pt.z*pt.y;
            cov(2,2) += pt.z*pt.z;
        }

        mean /= static_cast<double> (kCorrespondences_);
        for (int k = 0; k < 3; k++) {
            for (int l =0; l <= k; l++) {
                cov(k,l) /= static_cast<double> (kCorrespondences_);
                cov(k,l) -= mean[k]*mean[l];
                cov(l,k) = cov(k,l);
            }
        }

        // SVD decomposition for PCA
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
        cov.setZero();
        Eigen::Matrix3d U = svd.matrixU();

        for (int k = 0; k<3; k++) {
            Eigen::Vector3d col = U.col(k);
            double v = 1.;
            if (k == 2)
                v = epsilon_;
            cov+= v*col*col.transpose();
        }
        (*matvecptr)[itter] = cov;
    }

}

} // namespace semanticicp

#endif //GICP_HPP_
