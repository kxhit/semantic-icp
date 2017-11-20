#ifndef SEMANTIC_ICP_HPP_
#define SEMANTIC_ICP_HPP_

#include <iostream>

#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

#include <gicp_cost_functor_autodiff.h>
#include <local_parameterization_se3.h>

#include<Eigen/StdVector>

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
    //std::cout << "Init Transform\n" << initTransform.matrix() << std::endl;
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
            //if (sourceCloud_->labeledPointClouds[s]->size()>1400) {
            if (sourceCloud_->labeledPointClouds[s]->size()>400) {
            typename pcl::PointCloud<PointT>::Ptr transformedSource (new pcl::PointCloud<PointT>());
            Sophus::SE3d transform = currentTransform;
            Eigen::Matrix4d transMat = transform.matrix();
            // std::cout << "init transform\n" << transMat;
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
                //if( true ) {
                    const PointT &sourcePoint =
                        (sourceCloud_->labeledPointClouds[s])->points[sourceIndx];
                    const Eigen::Matrix3d &sourceCov =
                        (sourceCloud_->labeledCovariances[s])->at(sourceIndx);
                    const PointT &targetPoint =
                        (targetCloud_->labeledPointClouds[s])->points[targetIndx[0]];
                    const Eigen::Matrix3d &targetCov =
                        (targetCloud_->labeledCovariances[s])->at(targetIndx[0]);

                    //if(std::isnan(targetCov(1,1))){
                    //    std::cout << "Sqdist: " << distSq[0]<<std::endl;
                    //    std::cout << sourcePoint << std::endl;
                    //    std::cout << sourceCov << std::endl;
                    //    std::cout << targetPoint << std::endl;
                    //    std::cout << targetCov << std::endl;
                    //    std::cout << transformedSourcePoint << std::endl;
                    //}

                    GICPCostFunctorAutoDiff *c= new GICPCostFunctorAutoDiff(sourcePoint,
                                                                           targetPoint,
                                                                           sourceCov,
                                                                           targetCov,
                                                                           baseTransformation_);
                    ceres::CostFunction* cost_function =
                        new ceres::AutoDiffCostFunction<GICPCostFunctorAutoDiff,
                                                        1,
                                                        Sophus::SE3d::num_parameters>(c);

                    problem.AddResidualBlock(cost_function, new ceres::CauchyLoss(0.5),
                                             estTransform.data());
                }

            }
            /*
            if( problem.NumResidualBlocks()>3 ) {
            // Get Covariance
            ceres::Covariance::Options covOptions;
            ceres::Covariance cov(covOptions);

            std::vector<std::pair<const double*, const double*>> covBlocks;
            covBlocks.push_back(std::make_pair(estTransformation.data(),
                                               estTransformation.data()));

            bool goodCov = cov.Compute(covBlocks, &problem);
            Eigen::Matrix<double, 6,6> covMat;
            double temp[6*6];
            if ( goodCov )
                cov.GetCovarianceBlockInTangentSpace(estTransformation.data(),
                                                     estTransformation.data(),
                                                     temp);
            covMat = Eigen::Map<Eigen::Matrix<double, 6, 6, Eigen::RowMajor>>(temp);
            covMat = covMat;
            //covMat = Eigen::Matrix<double, 6, 6>::Identity();


            // Print Results
            std::cout << summary.BriefReport() << std::endl;
            double mse = (currentTransform.inverse()*estTransformation).log().squaredNorm();
            std::cout << "label transform squared difference: " << mse << std::endl;

            if( summary.IsSolutionUsable() && goodCov ) {
                transformsVec.push_back(estTransformation);
                covVec.push_back(covMat);
            }
            }
            */
            }
            }

        }
        // Sovler Options
        ceres::Solver::Options options;
        options.gradient_tolerance = 0.1 * Sophus::Constants<double>::epsilon();
        options.function_tolerance = 0.1 * Sophus::Constants<double>::epsilon();
        options.linear_solver_type = ceres::DENSE_QR;
        options.num_threads = 4;
        options.max_num_iterations = 400;

        // Solve
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        /*
        //Sophus::SE3d newTransform = iterativeMean(transformsVec,20);
        Eigen::Matrix4d ident = Eigen::Matrix4d::Identity();
        Sophus::SE3d identity(ident);
        Sophus::SE3d newTransform = poseFusion(transformsVec, covVec, currentTransform);
        */
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
