#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/console/parse.h>

#include <semantic_point_cloud.h>
#include <semantic_icp.h>
#include <semantic_viewer.h>
#include <pcl_2_semantic.h>


int
main (int argc, char** argv)
{
    std::string strSource;
    std::string strTarget;
    if ( !pcl::console::parse_argument(argc, argv, "-s", strSource) ) {
        std::cout << "Need source file (-s)\n";
        return (-1);
    }
    if ( !pcl::console::parse_argument(argc, argv, "-t", strTarget) ) {
        std::cout << "Need target file (-t)\n";
        return (-1);
    }

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloudA (new pcl::PointCloud<pcl::PointXYZL>);

    if (pcl::io::loadPCDFile<pcl::PointXYZL> (strSource, *cloudA) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file cloudA.pcd \n");
        return (-1);
    }
    std::cout << "Loaded "
              << cloudA->width * cloudA->height
              << " data points from cloudA.pcd with the following fields: "
              << std::endl;

    std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
        semanticAfinal (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

    semanticicp::pcl_2_semantic(cloudA, semanticAfinal);
    std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
        semanticA (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

    semanticicp::pcl_2_semantic(cloudA, semanticA);
    semanticA->removeSemanticClass( 3 );
    semanticA->removeSemanticClass( 10 );
    semanticA->removeSemanticClass( 11 );

    pcl::PointCloud<pcl::PointXYZL>::Ptr cloudB (new pcl::PointCloud<pcl::PointXYZL>);

    if (pcl::io::loadPCDFile<pcl::PointXYZL> (strTarget, *cloudB) == -1) //* load the file
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
    semanticB->removeSemanticClass( 3 );
    semanticB->removeSemanticClass( 10 );
    semanticB->removeSemanticClass( 11 );

    semanticicp::SemanticIterativeClosestPoint<pcl::PointXYZ, uint32_t> sicp;
    sicp.setInputSource(semanticA);
    sicp.setInputTarget(semanticB);

    auto begin = std::chrono::steady_clock::now();
    sicp.align(semanticAfinal);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Time Multiclass: "
              << std::chrono::duration_cast<std::chrono::seconds>(end-begin).count() << std::endl;

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

    begin = std::chrono::steady_clock::now();
    sicp2.align(semanticAnoL);
    end = std::chrono::steady_clock::now();
    std::cout << "Time Single Class: "
              << std::chrono::duration_cast<std::chrono::seconds>(end-begin).count() << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudAnoL (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ> (strSource, *cloudAnoL);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudBnoL (new pcl::PointCloud<pcl::PointXYZ>);
    pcl::io::loadPCDFile<pcl::PointXYZ> (strTarget, *cloudBnoL);

    pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZL, pcl::PointXYZL> gicp;
    gicp.setInputCloud(cloudA);
    gicp.setInputTarget(cloudB);
    pcl::PointCloud<pcl::PointXYZL>::Ptr final1(new pcl::PointCloud<pcl::PointXYZL>());
    gicp.align(*final1);

    std::cout << "GICP transform: \n" << gicp.getFinalTransformation() << std::endl;

    std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
        gicpFinal (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

    semanticicp::pcl_2_semantic(final1, gicpFinal);
    semanticicp::SemanticViewer<pcl::PointXYZ, uint32_t> viewer;
    viewer.addSemanticPointCloudSingleColor( semanticB, 231,41,138, "Target");
    viewer.addSemanticPointCloudSingleColor( semanticAfinal, 27,158,119, "GICP");
    viewer.addSemanticPointCloudSingleColor( gicpFinal, 217,95,2, "Semantic ICP");

    while(!viewer.wasStopped()) {
        std::this_thread::sleep_for (std::chrono::microseconds (100000));
    };

    return (0);
}
