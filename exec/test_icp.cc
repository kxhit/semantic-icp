#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include <semantic_point_cloud.h>
#include <semantic_icp.h>
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
              << " data points from cloudA.pcd with the following fields: "
              << std::endl;

    std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
        semanticA (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

    semanticicp::pcl_2_semantic(cloud, semanticA);

    std::cout << "LabelsA: " << std::endl;
    for(uint32_t l: semanticA->semanticLabels) {
        std::cout << l << std::endl;
    }

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloudB (new pcl::PointCloud<pcl::PointXYZL>);

    if (pcl::io::loadPCDFile<pcl::PointXYZL> ("cloudB.pcd", *cloudB) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file cloudB.pcd \n");
        return (-1);
    }
    std::cout << "Loaded "
              << cloudB->width * cloudB->height
              << " data points from cloudB.pcd with the following fields: "
              << std::endl;

    std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
        semanticB (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

    semanticicp::pcl_2_semantic(cloudB, semanticB);

    std::cout << "LabelsB: " << std::endl;
    for(uint32_t l: semanticB->semanticLabels) {
        std::cout << l << std::endl;
    }

    semanticicp::SemanticIterativeClosestPoint<pcl::PointXYZ, uint32_t> sicp;
    sicp.setInputSource(semanticA);
    sicp.setInputTarget(semanticB);

    sicp.align(semanticA);



    return (0);
}
