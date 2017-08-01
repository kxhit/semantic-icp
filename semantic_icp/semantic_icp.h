#ifndef SEMANTIC_ICP_H_
#define SEMANTIC_ICP_H_

#include <pcl/registration/icp.h>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/types.hpp>
#include <sophus/common.hpp>

#include <semantic_point_cloud.h>

namespace semanticicp
{
    template <typename PointT, typename SemanticT>
    class SemanticIterativeClosestPoint
    {
        public:
        typedef SemanticPointCloud<PointT, SemanticT> SemanticCloud;
        typedef typename std::shared_ptr<SemanticCloud> SemanticCloudPtr;
        typedef typename std::shared_ptr< const SemanticCloud> SemanticCloudConstPtr;

        typedef std::vector< Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > MatricesVector;
        typedef std::shared_ptr< MatricesVector > MatricesVectorPtr;
        typedef std::shared_ptr< const MatricesVector > MatricesVectorConstPtr;

        typedef pcl::KdTreeFLANN<PointT> KdTree;
        typedef typename KdTree::Ptr KdTreePtr;

        typedef Eigen::Matrix<double, 6, 1> Vector6d;

        SemanticIterativeClosestPoint()
        {
            Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
            baseTransformation_ = Sophus::SE3d(mat);
        };

        inline void
        setInputSource( const SemanticCloudPtr &cloud ) {
            sourceCloud_ = cloud;
        };

        inline void
        setInputTarget ( const SemanticCloudPtr &cloud ) {
            targetCloud_ = cloud;
        };


        void
        align(SemanticCloudPtr final);

        void
        align(SemanticCloudPtr final, Sophus::SE3d &initTransform);


        protected:

        int kCorrespondences_;
        double translationEpsilon_;
        double rotationEpsilon_;
        int maxInnerIterations_;

        Sophus::SE3d baseTransformation_;
        Sophus::SE3d finalTransformation_;

        SemanticCloudPtr sourceCloud_;
        SemanticCloudPtr targetCloud_;

        Sophus::SE3d iterativeMean( std::vector<Sophus::SE3d> const& in, size_t maxIterations);

    };

} // namespace semanticicp

#include <impl/semantic_icp.hpp>

#endif // #ifndef SEMANTIC_ICP_H_
