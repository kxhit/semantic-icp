#ifndef SEMANTIC_POINT_CLOUD_H_
#define SEMANTIC_POINT_CLOUD_H_

#include <vector>
#include <algorithm>
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
        typedef std::shared_ptr<SemanticPointCloud<PointT, SemanticT>> Ptr;
        typedef std::shared_ptr< const SemanticPointCloud<PointT, SemanticT>> ConstPtr;

        typedef pcl::PointCloud<PointT> PointCloud;
        typedef typename PointCloud::Ptr PointCloudPtr;

        typedef pcl::KdTreeFLANN<PointT> KdTree;
        typedef typename KdTree::Ptr KdTreePtr;

        typedef std::vector< Eigen::Matrix3d, Eigen::aligned_allocator<Eigen::Matrix3d> > MatricesVector;
        typedef std::shared_ptr<MatricesVector> MatricesVectorPtr;

        SemanticPointCloud(int k=20, double epsilon = 0.001) :
        k_correspondences_(k),
        epsilon_(epsilon)
        {};

        std::vector<SemanticT> semanticLabels;
        std::map<SemanticT, PointCloudPtr> labeledPointClouds;
        std::map<SemanticT, MatricesVectorPtr> labeledCovariances;
        std::map<SemanticT, KdTreePtr> labeledKdTrees;

        void addSemanticCloud( SemanticT label, PointCloudPtr cloud_ptr,
                bool computeKd = true, bool computeCov = true);

        void removeSemanticClass( SemanticT label ) {
            auto it = std::find( semanticLabels.begin(), semanticLabels.end(), label );
            if( it != semanticLabels.end() ) {
                semanticLabels.erase(it);
                labeledPointClouds.erase(labeledPointClouds.find(label));
                labeledCovariances.erase(labeledCovariances.find(label));
                labeledKdTrees.erase(labeledKdTrees.find(label));
            }
        };

        pcl::PointCloud<pcl::PointXYZL>::Ptr getpclPointCloud ();

        void transform(Eigen::Matrix4f trans);

        private:

        int k_correspondences_;
        double epsilon_;

    };
} // namespace semanticicp

#include <impl/semantic_point_cloud.hpp>

#endif //SEMANTIC_POINT_CLOUD_H_
