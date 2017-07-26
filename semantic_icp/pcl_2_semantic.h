#ifndef PCL_2_SEMANTIC_H_
#define PCL_2_SEMANTIC_H_

#include <vector>
#include <memory>
#include <map>


#include <semantic_point_cloud.h>

namespace semanticicp
{

    void
    pcl_2_semantic(const pcl::PointCloud<pcl::PointXYZL>::Ptr pclCloud,
                   std::shared_ptr<SemanticPointCloud<pcl::PointXYZ, uint32_t>> semanticCloud) {

        typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;
        typedef typename PointCloud::Ptr PointCloudPtr;

        std::vector<uint32_t> labels;
        std::map<uint32_t, PointCloudPtr> map;

        for(pcl::PointXYZL p:pclCloud->points) {
            if(map.find(p.label) == map.end()){
                PointCloudPtr cloud ( new PointCloud());
                cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
                map[p.label] = cloud;
                labels.push_back(p.label);
            }
            else {
                PointCloudPtr cloud = map[p.label];
                cloud->push_back(pcl::PointXYZ(p.x, p.y, p.z));
            }
        }

        for( uint32_t l: labels) {
            PointCloudPtr cloud = map[l];
            semanticCloud->addSemanticCloud(l, cloud);
        }

    };

} // namespace semanticicp

#endif //PCL_2_SEMANTIC_H_
