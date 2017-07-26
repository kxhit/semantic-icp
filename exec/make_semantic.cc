#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <semantic_point_cloud.h>
#include <pcl_2_semantic.h>

int
main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZL>);

    if (pcl::io::loadPCDFile<pcl::PointXYZL> ("cloudA.pcd", *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file cloudA.pcd \n");
        return (-1);
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
        std::cout << l << std::endl;
    }

    return (0);
}
