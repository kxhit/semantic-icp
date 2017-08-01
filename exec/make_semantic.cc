#include <iostream>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <semantic_point_cloud.h>
#include <pcl_2_semantic.h>

int
main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL>);

    if (pcl::io::loadPCDFile<pcl::PointXYZL> ("cloudB.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file cloudA.pcd \n");
        return (-1);
    }

    for(size_t t = 0; t< cloud->points.size(); t++){
        pcl::PointXYZL p = cloud->points[t];
        p.label=0;
        cloud->points[t] = p;
    }
    std::cout << "Loaded "
              << cloud->width * cloud->height
              << " data points from test_pcd.pcd with the following fields: "
              << std::endl;

    std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
        semanticCloud (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

    semanticicp::pcl_2_semantic(cloud, semanticCloud);

    std::cout << "Labels: " << std::endl;
    for(uint32_t l: semanticCloud->semanticLabels) {
        semanticicp::SemanticPointCloud<pcl::PointXYZL, uint32_t>::MatricesVectorPtr mVec =
            semanticCloud->labeledCovariances[l];
        for(Eigen::Matrix3d m: *mVec) {
            if(std::isnan(m(1,1)))
            std::cout << m << std::endl;
        }
    }

    return (0);
}
