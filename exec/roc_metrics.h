#ifndef _ROC_METRICS_H_
#define _ROC_METRICS_H_

#include <ostream>

class ROCMetrics
{
    public:
        typedef pcl::PointXYZL PointT;

        typedef pcl::PointCloud<PointT> PointCloud;
        typedef typename PointCloud::Ptr PointCloudPtr;

        typedef pcl::KdTreeFLANN<PointT> KdTree;
        typedef typename KdTree::Ptr KdTreePtr;

        ROCMetrics( std::ostream *out ) {
            out_ = out;
        };

        void
        evaluate(PointCloudPtr source,
                 PointCloudPtr target) {

            KdTreePtr tree(new KdTree());
            tree->setInputCloud(target);

            for(PointT p: *source) {
                std::vector<int> nn_index; nn_index.reserve (1);
                std::vector<float> nn_dist_sq; nn_dist_sq.reserve (1);

                tree->nearestKSearch(p, 1, nn_index, nn_dist_sq);

                if ( nn_dist_sq[0]<25.0 ) {
                    uint32_t labelSource = p.label;
                    uint32_t labelTarget = target->at(nn_index[0]).label;

                    *out_ << labelSource << ", " << labelTarget << std::endl;
                }
            }
        }

    private:
        std::ostream *out_;
};

#endif // _ROC_METRICS_H_
