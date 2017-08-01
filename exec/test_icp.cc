#include <iostream>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>

#include <semantic_point_cloud.h>
#include <semantic_icp.h>
#include <pcl_2_semantic.h>

int
main (int argc, char** argv)
{
    pcl::PointCloud<pcl::PointXYZL>::Ptr cloudA (new pcl::PointCloud<pcl::PointXYZL>);

    if (pcl::io::loadPCDFile<pcl::PointXYZL> ("cloudA.pcd", *cloudA) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file cloudA.pcd \n");
        return (-1);
    }
    std::cout << "Loaded "
              << cloudA->width * cloudA->height
              << " data points from cloudA.pcd with the following fields: "
              << std::endl;

    std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
        semanticA (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

    semanticicp::pcl_2_semantic(cloudA, semanticA);

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

    semanticicp::SemanticIterativeClosestPoint<pcl::PointXYZ, uint32_t> sicp;
    sicp.setInputSource(semanticA);
    sicp.setInputTarget(semanticB);

    sicp.align(semanticA);

    for(size_t t = 0; t< cloudA->points.size(); t++){
        pcl::PointXYZL p = cloudA->points[t];
        p.label=0;
        cloudA->points[t] = p;
    }
    std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
        semanticAnoL (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

    std::cout << "LabelsA: " << std::endl;
    semanticicp::pcl_2_semantic(cloudA, semanticAnoL);

    for(uint32_t l: semanticAnoL->semanticLabels) {
        std::cout << l << " Num Points: " <<semanticAnoL->labeledPointClouds[l]->points.size()
                  <<std::endl;
    }

    for(size_t t = 0; t< cloudB->points.size(); t++){
        pcl::PointXYZL p = cloudB->points[t];
        p.label=0;
        cloudB->points[t] = p;
    }
    std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
        semanticBnoL (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

    semanticicp::pcl_2_semantic(cloudB, semanticBnoL);
    std::cout << "LabelsB: " << std::endl;
    for(uint32_t l: semanticBnoL->semanticLabels) {
        std::cout << l << " Num Points: " <<semanticBnoL->labeledPointClouds[l]->points.size()
                  <<std::endl;
    }

    semanticicp::SemanticIterativeClosestPoint<pcl::PointXYZ, uint32_t> sicp2;
    sicp2.setInputSource(semanticAnoL);
    sicp2.setInputTarget(semanticBnoL);

    sicp2.align(semanticAnoL);
    /*
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudAnoL (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ> ("cloudA.pcd", *cloudAnoL);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudBnoL (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ> ("cloudB.pcd", *cloudBnoL);

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
    gicp.setInputCloud(cloudAnoL);
    gicp.setInputTarget(cloudBnoL);
    pcl::PointCloud<pcl::PointXYZ> final1;
    gicp.align(final1);

    std::cout << "GICP transform: \n" << gicp.getFinalTransformation() << std::endl;
    */

    return (0);
}
