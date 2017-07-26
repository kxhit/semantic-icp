#ifndef SEMANTIC_POINT_CLOUD_H_
#define SEMANTIC_POINT_CLOUD_H_

#include <vector>
#include <memory>
#include <map>

#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>


namespace semanticicp
{
    template <typename PointT, typename SemanticT>
    class SemanticPointCloud
    {
        public:
        typedef pcl::PointCloud<PointT> PointCloud;
        typedef typename PointCloud::Ptr PointCloudPtr;

        typedef pcl::KdTreeFLANN<PointT> KdTree;
        typedef typename KdTree::Ptr KdTreePtr;

        typedef std::vector< Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > MatricesVector;
        typedef std::shared_ptr<MatricesVector> MatricesVectorPtr;

        SemanticPointCloud(int k=20, double epsilon = 0.001) :
        k_correspondences_(k)
        {};

        std::vector<SemanticT> semanticLabels;
        std::map<SemanticT, PointCloudPtr> labeledPointClouds;
        std::map<SemanticT, MatricesVectorPtr> labeledCovariances;
        std::map<SemanticT, KdTreePtr> labeledKdTrees;

        void addSemanticCloud( SemanticT label, PointCloudPtr cloud_ptr,
                bool computeKd = true, bool computeCov = true);

        private:

        int k_correspondences_;
        double epsilon_;

    };
} // namespace semanticicp

#include <impl/semantic_point_cloud.hpp>

#endif //SEMANTIC_POINT_CLOUD_H_
