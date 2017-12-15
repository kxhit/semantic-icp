// Copyright 2017 Steven Parkison

#ifndef SEMANTIC_ICP_IMPL_EM_ICP_HPP_
#define SEMANTIC_ICP_IMPL_EM_ICP_HPP_

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


namespace semanticicp {

template <size_t N>
void EmIterativeClosestPoint<N>::align(PointCloudPtr final_cloud,
                                    const Sophus::SE3d & init_transform) {

  ComputeCovariances(source_cloud_, source_kd_tree_, source_covariances_, source_distributions_);
  ComputeCovariances(target_cloud_, target_kd_tree_, target_covariances_, target_distributions_);
  Sophus::SE3d current_transform(init_transform);
  bool converged = false;
  size_t outter_itter = 0;

  while(converged!=true) {

    // Build The Problem
    ceres::Problem problem;

    // Add Sophus SE3 Parameter block with local parametrization
    Sophus::SE3d est_transform(current_transform);
    problem.AddParameterBlock(est_transform.data(), Sophus::SE3d::num_parameters,
                              new LocalParameterizationSE3);

    double mse_high = 0;

    typename pcl::PointCloud<PointT>::Ptr transformed_source (new pcl::PointCloud<PointT>());
    Eigen::Matrix4d trans_mat = current_transform.matrix();
    pcl::transformPointCloud(*source_cloud_,
                                *transformed_source,
                                trans_mat);

    std::vector<int> target_index;
    std::vector<float> dist_sq;


    std::cout << "Num Points: " << transformed_source->size() << std::endl;
    for(int source_index = 0; source_index != transformed_source->size(); source_index++) {
      const PointT &transformed_source_pt = transformed_source->points[source_index];

      target_kd_tree_->nearestKSearch(transformed_source_pt, 8,
                                      target_index, dist_sq);
      for(int correspondence_index = 0;
          correspondence_index < 8;
          correspondence_index++) {
        if( dist_sq[correspondence_index] < 250 ) {
          const PointT &source_pt =
            source_cloud_->points[source_index];
          const pcl::PointXYZ s_pt(source_pt.x, source_pt.y, source_pt.z);
          const Eigen::Matrix3d &source_cov =
            source_covariances_->at(source_index);
          const PointT &target_pt =
            target_cloud_->points[target_index[correspondence_index]];
          const pcl::PointXYZ t_pt(target_pt.x, target_pt.y, target_pt.z);
          const Eigen::Matrix3d &target_cov =
            target_covariances_->at(target_index[correspondence_index]);

          const Eigen::Matrix<double,N, 1> dist =
            target_distributions_->at(target_index[correspondence_index]);

          double prob = confusion_matrix_(source_pt.label-1, target_pt.label-1)*
                        dist(target_pt.label-1, 0);

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
          ceres::CostFunction* cost_function = new GICPCostFunction(s_pt,
                                                                    t_pt,
                                                                    source_cov,
                                                                    target_cov,
                                                                    base_transformation_);
          problem.AddResidualBlock(cost_function,
                                   new ceres::ScaledLoss(new ceres::CauchyLoss(1.5),
                                                         prob,
                                                         ceres::TAKE_OWNERSHIP),
                                   est_transform.data());

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
            params.push_back(est_transform.data());
            if (!gradient_checker.Probe(params.data(), 5e-4, &results)) {
              std::cout << "An error has occurred:\n";
              std::cout << results.error_log;
              std::cout << results.jacobians[0] << std::endl;
              std::cout << results.numeric_jacobians[0] << std::endl;
              std::cout << est_transform.matrix() << std::endl;
              std::cout << source_pt << std::endl;
              std::cout << target_pt << std::endl;
              std::cout << source_cov << std::endl;
              std::cout << target_cov << std::endl;
            }  // Gradient Relitive error
          }  // Gradient Check
        }  // If close enough
      }  // For loop over correspondences
    }  // For loop over points
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

    double mse = (current_transform.inverse()*est_transform).log().squaredNorm();
    if(mse < 0.001 || outter_itter>50)
        converged = true;
    std::cout<< "MSE: " << mse << std::endl;
    std::cout<< "Transform: " << std::endl;
    std::cout<< est_transform.matrix() << std::endl;
    std::cout<< "Itteration: " << outter_itter << std::endl;
    current_transform = est_transform;
    outter_itter++;
  }

  final_transformation_ = current_transform;

  Sophus::SE3d trans = final_transformation_*base_transformation_;
  Eigen::Matrix4f mat = (trans.matrix()).cast<float>();
  if( final_cloud != nullptr ) {
      pcl::transformPointCloud(*source_cloud_,
                               *final_cloud,
                               mat);
  }

}

template <size_t N>
void EmIterativeClosestPoint<N>::ComputeCovariances(
    const PointCloudPtr cloudptr,
    KdTreePtr treeptr,
    MatricesVectorPtr matvecptr,
    DistVectorPtr distvecptr) {
  // Variables for computing Covariances
  Eigen::Vector3d mean;
  Eigen::Matrix<double, N, 1> dist;
  double increment = 1.0/static_cast<double>(kCorrespondences_);

  std::vector<int> nn_idecies; nn_idecies.reserve (kCorrespondences_);
  std::vector<float> nn_dist_sq; nn_dist_sq.reserve (kCorrespondences_);

  // Set up Itteration
  matvecptr->resize(cloudptr->size());
  distvecptr->resize(cloudptr->size());

  for(size_t itter = 0; itter < cloudptr->size(); itter++) {
    const PointT &query_pt = (*cloudptr)[itter];

    Eigen::Matrix3d cov;
    cov.setZero();
    mean.setZero();
    dist.setZero();

    treeptr->nearestKSearch(query_pt, kCorrespondences_, nn_idecies, nn_dist_sq);

    for( int index: nn_idecies) {
      const PointT &pt = (*cloudptr)[index];

      dist(pt.label-1,0) += increment;

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
      if (k == 2) {
        v = kEpsilon_;
      }
      cov+= v*col*col.transpose();
    }
    (*matvecptr)[itter] = cov;
    (*distvecptr)[itter] = dist;
  }


}
} // namespace semanticicp

#endif  // SEMANTIC_ICP_IMPL_EM_ICP_HPP_
