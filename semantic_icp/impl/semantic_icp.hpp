#ifndef SEMANTIC_ICP_HPP_
#define SEMANTIC_ICP_HPP_

#include <iostream>

#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

#include <gicp_cost_functor_autodiff.h>
#include <local_parameterization_se3.h>

namespace semanticicp
{

template <typename PointT, typename SemanticT>
void SemanticIterativeClosestPoint<PointT,SemanticT>::align(SemanticCloudPtr final) {
    Sophus::SE3d init;
    align(final, init);
};

template <typename PointT, typename SemanticT>
void SemanticIterativeClosestPoint<PointT,SemanticT>::align(
        SemanticCloudPtr finalCloud, Sophus::SE3d &initTransform) {
    Sophus::SE3d currentTransform(initTransform);
    bool converged = false;
    size_t count = 0;

    while(converged!=true) {
        std::vector<Sophus::SE3d> transformsVec;
        double mseHigh = 0;
        count++;
        for(SemanticT s:sourceCloud_->semanticLabels) {
            std::cout << "Label: " << s << std::endl;
            if (targetCloud_->labeledPointClouds.find(s) != targetCloud_->labeledPointClouds.end()) {
            typename pcl::PointCloud<PointT>::Ptr transformedSource (new pcl::PointCloud<PointT>());
            Sophus::SE3d transform = Sophus::SE3d(currentTransform*baseTransformation_);
            Eigen::Matrix4d transMat = transform.matrix();
            pcl::transformPointCloud(*(sourceCloud_->labeledPointClouds[s]),
                                    *transformedSource,
                                    transMat);

            KdTreePtr tree = targetCloud_->labeledKdTrees[s];
            std::vector<int> targetIndx;
            std::vector<float> distSq;

            // Build The Problem
            ceres::Problem problem;

            // Add Sophus SE3 Parameter block with local parametrization
            Sophus::SE3d estTransformation(currentTransform);
            problem.AddParameterBlock(estTransformation.data(), Sophus::SE3d::num_parameters,
                                      new LocalParameterizationSE3);


            for(int sourceIndx = 0; sourceIndx != transformedSource->size(); sourceIndx++) {
                const PointT &sourcePoint = transformedSource->points[sourceIndx];

                tree->nearestKSearch(sourcePoint, 1, targetIndx, distSq);
                if( distSq[0] < 4 ) {
                    const Eigen::Matrix3d &sourceCov =
                        (sourceCloud_->labeledCovariances[s])->at(sourceIndx);
                    const PointT &targetPoint =
                        (targetCloud_->labeledPointClouds[s])->points[targetIndx[0]];
                    const Eigen::Matrix3d &targetCov =
                        (targetCloud_->labeledCovariances[s])->at(targetIndx[0]);

                    if(std::isnan(targetCov(1,1))){
                        std::cout << "Invalid set: \n";
                        std::cout << sourcePoint << std::endl;
                        std::cout << sourceCov << std::endl;
                        std::cout << targetPoint << std::endl;
                        std::cout << targetCov << std::endl;
                    }

                    GICPCostFunctorAutoDiff *c= new GICPCostFunctorAutoDiff(sourcePoint,
                                                                           targetPoint,
                                                                           sourceCov,
                                                                           targetCov,
                                                                           baseTransformation_);
                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<GICPCostFunctorAutoDiff,
                                                        1,
                                                        Sophus::SE3d::num_parameters>(c);

                    problem.AddResidualBlock(cost_function, NULL, estTransformation.data());
                }

            }
            // Sovler Options
            ceres::Solver::Options options;
            //options.gradient_tolerance = 0.1 * Sophus::Constants<double>::epsilon();
            //options.function_tolerance = 0.1 * Sophus::Constants<double>::epsilon();
            options.linear_solver_type = ceres::DENSE_QR;

            // Solve
            ceres::Solver::Summary summary;
            ceres::Solve(options, &problem, &summary);

            // Print Results
            std::cout << summary.BriefReport() << std::endl;
            double mse = (currentTransform.inverse()*estTransformation).log().squaredNorm();
            std::cout << "label transform squared difference: " << mse << std::endl;

            if(summary.IsSolutionUsable())
                transformsVec.push_back(estTransformation);

            }

        }
        Sophus::SE3d newTransform = iterativeMean(transformsVec,20);
        double mse = (currentTransform.inverse()*newTransform).log().squaredNorm();
        if(mse < 0.001 || count>20)
            converged = true;
        std::cout<< "MSE: " << mse << std::endl;
        std::cout<< "Transform: " << std::endl;
        std::cout<< newTransform.matrix() << std::endl;
        currentTransform = newTransform;
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

} // namespace semanticicp

#endif //SEMANTIC_ICP_HPP_
