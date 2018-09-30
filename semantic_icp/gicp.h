#ifndef GICP_H_
#define GICP_H_

#include <pcl/registration/icp.h>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/types.hpp>
#include <sophus/common.hpp>

#include <semantic_point_cloud.h>

namespace semanticicp
{
    template <typename PointT>
    class GICP
    {
        public:
        typedef pcl::PointCloud<PointT> PointCloud;
        typedef typename PointCloud::Ptr PointCloudPtr;

        typedef std::vector< Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > MatricesVector;
        typedef std::vector< Eigen::Matrix<double,6,6>,
                             Eigen::aligned_allocator<Eigen::Matrix<double,6,6> > >
                                 CovarianceVector;

        typedef std::shared_ptr< MatricesVector > MatricesVectorPtr;
        typedef std::shared_ptr< const MatricesVector > MatricesVectorConstPtr;

        typedef pcl::KdTreeFLANN<PointT> KdTree;
        typedef typename KdTree::Ptr KdTreePtr;

        typedef Eigen::Matrix<double, 6, 1> Vector6d;

        GICP(int k = 20, double epsilon = 0.001) :
        kCorrespondences_(k),
        epsilon_(epsilon)
        {
            Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
            baseTransformation_ = Sophus::SE3d(mat);
        };

        inline void
        setSourceCloud( const PointCloudPtr &cloud ) {
            sourceCloud_ = cloud;
            sourceKdTree_ = KdTreePtr(new KdTree());
            sourceKdTree_->setInputCloud(sourceCloud_);
            sourceCovariances_ = MatricesVectorPtr(new MatricesVector());
        };

        inline void
        setSourceCloud( const PointCloudPtr &cloud, const KdTreePtr &tree, const MatricesVectorPtr &covs) {
            sourceCloud_ = cloud;
            sourceKdTree_ = tree;
            sourceCovariances_ = covs;
        };

        inline void
        setTargetCloud ( const PointCloudPtr &cloud ) {
            targetCloud_ = cloud;
            targetKdTree_ = KdTreePtr(new KdTree());
            targetKdTree_->setInputCloud(targetCloud_);
            targetCovariances_ = MatricesVectorPtr(new MatricesVector());
        };

        inline void
        setTargetCloud( const PointCloudPtr &cloud, const KdTreePtr &tree, const MatricesVectorPtr &covs) {
            targetCloud_ = cloud;
            targetKdTree_ = tree;
            targetCovariances_ = covs;
        };

        inline KdTreePtr
        getSourceKdTree() {
            return sourceKdTree_;
        };

        inline MatricesVectorPtr
        getSourceCovariances() {
            return sourceCovariances_;
        };

        inline KdTreePtr
        getTargetKdTree() {
            return targetKdTree_;
        };

        inline MatricesVectorPtr
        getTargetCovariances() {
            return targetCovariances_;
        };

        void
        align(PointCloudPtr finalCloud);

        void
        align(PointCloudPtr finalCloud, Sophus::SE3d &initTransform);

        Sophus::SE3d
        getFinalTransFormation() {
            Sophus::SE3d temp = finalTransformation_;
            return temp;
        };

        int
        getOuterIter(){
            return outer_iter;
        }

        protected:

        int kCorrespondences_;
        double epsilon_;
        double translationEpsilon_;
        double rotationEpsilon_;
        int maxInnerIterations_;

        int outer_iter;

        Sophus::SE3d baseTransformation_;
        Sophus::SE3d finalTransformation_;

        PointCloudPtr sourceCloud_;
        KdTreePtr sourceKdTree_;
        MatricesVectorPtr sourceCovariances_;

        PointCloudPtr targetCloud_;
        KdTreePtr targetKdTree_;
        MatricesVectorPtr  targetCovariances_;

        void computeCovariances(const PointCloudPtr cloudptr, KdTreePtr treeptr, MatricesVectorPtr matvec);

    };

} // namespace semanticicp

#include <impl/gicp.hpp>

#endif // #ifndef GICP_H_
