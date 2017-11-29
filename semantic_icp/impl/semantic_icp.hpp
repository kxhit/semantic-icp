#ifndef SEMANTIC_ICP_HPP_
#define SEMANTIC_ICP_HPP_

#include <iostream>

#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

//#include <gicp_cost_functor_autodiff.h>
#include <gicp_cost_function.h>
#include <local_parameterization_se3.h>

#include<Eigen/StdVector>
#include<ceres/gradient_checker.h>

namespace semanticicp
{

template <typename PointT, typename SemanticT>
void SemanticIterativeClosestPoint<PointT,SemanticT>::align(SemanticCloudPtr finalCloud) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    Sophus::SE3d init(mat);
    align(finalCloud, init);
};

template <typename PointT, typename SemanticT>
void SemanticIterativeClosestPoint<PointT,SemanticT>::align(
        SemanticCloudPtr finalCloud, Sophus::SE3d &initTransform) {
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
        count++;
        for(SemanticT s:sourceCloud_->semanticLabels) {
            std::cout << "Label: " << s << std::endl;
            if (targetCloud_->labeledPointClouds.find(s) != targetCloud_->labeledPointClouds.end()) {
            if (sourceCloud_->labeledPointClouds[s]->size()>400) {
            typename pcl::PointCloud<PointT>::Ptr transformedSource (new pcl::PointCloud<PointT>());
            Sophus::SE3d transform = currentTransform;
            Eigen::Matrix4d transMat = transform.matrix();
            pcl::transformPointCloud(*(sourceCloud_->labeledPointClouds[s]),
                                    *transformedSource,
                                    transMat);

            KdTreePtr tree = targetCloud_->labeledKdTrees[s];
            std::vector<int> targetIndx;
            std::vector<float> distSq;


            std::cout << "Num Points: " << transformedSource->size() << std::endl;
            for(int sourceIndx = 0; sourceIndx != transformedSource->size(); sourceIndx++) {
                const PointT &transformedSourcePoint = transformedSource->points[sourceIndx];

                tree->nearestKSearch(transformedSourcePoint, 1, targetIndx, distSq);
                if( distSq[0] < 250 ) {
                    const PointT &sourcePoint =
                        (sourceCloud_->labeledPointClouds[s])->points[sourceIndx];
                    const Eigen::Matrix3d &sourceCov =
                        (sourceCloud_->labeledCovariances[s])->at(sourceIndx);
                    const PointT &targetPoint =
                        (targetCloud_->labeledPointClouds[s])->points[targetIndx[0]];
                    const Eigen::Matrix3d &targetCov =
                        (targetCloud_->labeledCovariances[s])->at(targetIndx[0]);

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
                    ceres::CostFunction* cost_function = new GICPCostFunction(sourcePoint,
                                                                           targetPoint,
                                                                           sourceCov,
                                                                           targetCov,
                                                                           baseTransformation_);
                    problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(1.5),
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
            } // If (numpoints)
            } // If semantic is in target

        }
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
        if(mse < 0.001 || count>35)
            converged = true;
        std::cout<< "MSE: " << mse << std::endl;
        std::cout<< "Transform: " << std::endl;
        std::cout<< estTransform.matrix() << std::endl;
        std::cout<< "Itteration: " << count << std::endl;
        currentTransform = estTransform;
    }

    finalTransformation_ = currentTransform;

    Sophus::SE3d trans = finalTransformation_*baseTransformation_;
    Eigen::Matrix4f mat = (trans.matrix()).cast<float>();
    finalCloud->transform(mat);
};


template <typename PointT, typename SemanticT>
Sophus::SE3d SemanticIterativeClosestPoint<PointT,SemanticT>::iterativeMean(
        std::vector<Sophus::SE3d> const& in,
        size_t maxIterations) {
    size_t N = in.size();

    Sophus::SE3d tAverage = in.front();
    double w = double(1.0/N);
    for(size_t i = 0; i< maxIterations; ++i) {
        Sophus::SE3d::Tangent average;
        average.setZero();
        for(Sophus::SE3d const& transform: in) {
            average += w*(tAverage.inverse() * transform).log();
        }
        Sophus::SE3d newTAverage = tAverage*Sophus::SE3d::exp(average);
        if((newTAverage.inverse()*tAverage).log().squaredNorm()<0.01)
            return newTAverage;

        tAverage = newTAverage;
    }
    std::cout << "Iterative Mean Failed";
    return tAverage;
};

struct PoseFusionCostFunctor {
    PoseFusionCostFunctor(Sophus::SE3d pose,
                          Eigen::Matrix<double,6,6> cov) :
                          poseInv_(pose),
                          covInv_(cov) {}

    template <class T>
    bool operator()(T const* const parameters, T* residuals) const {
        Eigen::Map<Sophus::SE3<T> const> const testPose(parameters);

        Eigen::Matrix<T,6,1> res = (testPose*poseInv_.cast<T>()).log();
        residuals[0] = T(res.transpose()*covInv_.cast<T>()*res);
        return true;
    }

    Sophus::SE3d poseInv_;
    Eigen::Matrix<double,6,6> covInv_;
};

template <typename PointT, typename SemanticT>
Sophus::SE3d SemanticIterativeClosestPoint<PointT, SemanticT>::poseFusion(
        std::vector<Sophus::SE3d> const& poses,
        CovarianceVector const& covs,
        Sophus::SE3d const &initTransform) {

    for(size_t n =0; n<poses.size(); n++) {
        std::cout << poses[n].matrix() << std::endl;
        std::cout << covs[n] << std::endl;
    }

    if(poses.size() == 1)
        return poses[0];

    ceres::Problem problem;

    Sophus::SE3d fusedPose(initTransform);
    problem.AddParameterBlock(fusedPose.data(), Sophus::SE3d::num_parameters,
                              new LocalParameterizationSE3);

    double det = 0;
    for(auto m:covs) {
        det+=m.determinant()/double(covs.size());
    }
    double scale = pow(1.0/det, 1.0/6.0);
    scale = 1.0/scale;

    for(size_t n =0; n<poses.size(); n++) {
        //std::cout << poses[n].matrix() << std::endl;
        //std::cout << covs[n] << std::endl;
        PoseFusionCostFunctor *c = new PoseFusionCostFunctor(poses[n].inverse(),
                                                             covs[n].inverse()*scale);

        ceres::CostFunction *costFunction =
            new ceres::AutoDiffCostFunction<PoseFusionCostFunctor,
                                            1,
                                            Sophus::SE3d::num_parameters> (c);
        problem.AddResidualBlock(costFunction, new ceres::HuberLoss(10.0), fusedPose.data());
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.num_threads = 4;
    options.max_num_iterations = 50000;
    options.gradient_tolerance = 0.0001 * Sophus::Constants<double>::epsilon();
    options.function_tolerance = 0.0001 * Sophus::Constants<double>::epsilon();

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "Pose Fusion\n";
    std::cout << summary.BriefReport() << std::endl;

    return fusedPose;
};



} // namespace semanticicp

#endif //SEMANTIC_ICP_HPP_
