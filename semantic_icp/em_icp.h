// Copyright 2017 Steven Parkison

#ifndef SEMANTIC_ICP_EM_ICP_H_
#define SEMANTIC_ICP_EM_ICP_H_

#include <vector>

#include <sophus/se3.hpp>
#include <sophus/types.hpp>
#include <sophus/common.hpp>
#include <pcl/registration/icp.h>
#include <Eigen/Geometry>

namespace semanticicp {

template <size_t N>
class EmIterativeClosestPoint {
 public:
  typedef pcl::PointXYZL PointT;
  typedef typename pcl::PointCloud<PointT> PointCloud;
  typedef typename PointCloud::Ptr PointCloudPtr;

  typedef std::vector<Eigen::Matrix3d,
                      Eigen::aligned_allocator<Eigen::Matrix3d>>
                      MatricesVector;
  typedef std::vector<Eigen::Matrix<double, 6, 6>,
                      Eigen::aligned_allocator<Eigen::Matrix<double, 6, 6>>>
                      CovarianceVector;
  typedef std::vector<Eigen::Matrix<double, N, 1>,
                               Eigen::aligned_allocator<Eigen::Matrix<double, N, 1>>>
                               DistVector;

  typedef std::shared_ptr< MatricesVector > MatricesVectorPtr;
  typedef std::shared_ptr< const MatricesVector > MatricesVectorConstPtr;
  typedef std::shared_ptr< DistVector > DistVectorPtr;

  typedef typename pcl::KdTreeFLANN<PointT> KdTree;
  typedef typename KdTree::Ptr KdTreePtr;

  typedef Eigen::Matrix<double, 6, 1> Vector6d;

  EmIterativeClosestPoint(int k = 20,
                          double epsilon = 0.001) :
  kCorrespondences_(k),
  kEpsilon_(epsilon) {
    Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
    base_transformation_ = Sophus::SE3d(mat);
  }

  inline void
  setSourceCloud(const PointCloudPtr &cloud ) {
    source_cloud_ = cloud;
    source_kd_tree_ = KdTreePtr(new KdTree());
    source_kd_tree_->setInputCloud(source_cloud_);
    source_covariances_ = MatricesVectorPtr(new MatricesVector());
    source_distributions_ = DistVectorPtr(new DistVector());
  }

  inline void
  setTargetCloud(const PointCloudPtr &cloud ) {
    target_cloud_ = cloud;
    target_kd_tree_ = KdTreePtr(new KdTree());
    target_kd_tree_->setInputCloud(target_cloud_);
    target_covariances_ = MatricesVectorPtr(new MatricesVector());
    target_distributions_ = DistVectorPtr(new DistVector());
  }

  inline void
  setConfusionMatrix(const Eigen::Matrix<double, N, N> &in) {
    confusion_matrix_ = in;
  }

  void
  align(PointCloudPtr finalCloud);

  void
  align(PointCloudPtr finalCloud, const Sophus::SE3d &initTransform);

  void
  getFusedLabels(PointCloudPtr labeledCloud, const Sophus::SE3d &transformation);

  Sophus::SE3d
  getFinalTransFormation() {
    Sophus::SE3d temp = final_transformation_;
    return temp;
  }

  int
  getOuterIter() {
      return outer_iter;
  }

 protected:
  int kNumClasses_;
  int kCorrespondences_;
  double kEpsilon_;
  double kTranslationEpsilon_;
  double kRotationEpsilon_;
  int kMaxInnerIterations_;

  int outer_iter;

  Sophus::SE3d base_transformation_;
  Sophus::SE3d final_transformation_;

  PointCloudPtr source_cloud_;
  KdTreePtr source_kd_tree_;
  MatricesVectorPtr source_covariances_;
  DistVectorPtr source_distributions_;

  PointCloudPtr target_cloud_;
  KdTreePtr target_kd_tree_;
  MatricesVectorPtr  target_covariances_;
  DistVectorPtr target_distributions_;

  Eigen::Matrix<double, N, N> confusion_matrix_;

  void ComputeCovariances(const PointCloudPtr cloudptr,
                          KdTreePtr treeptr,
                          MatricesVectorPtr matvec,
                          DistVectorPtr distvec);
};

}  // namespace semanticicp

#include <impl/em_icp.hpp>

#endif  // SEMANTIC_ICP_EM_ICP_H_
