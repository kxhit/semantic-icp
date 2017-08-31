#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <string>
#include <dirent.h>
#include <algorithm>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/console/parse.h>

#include <semantic_point_cloud.h>
#include <semantic_icp.h>
#include <pcl_2_semantic.h>
#include "kitti_metrics.h"



std::vector<std::string>
get_pcd_in_dir(std::string dir_name) {

    DIR           *d;
    struct dirent *dir;
    d = opendir(dir_name.c_str());

    std::vector<std::string> pcd_fns;

    if (d) {
        while ((dir = readdir(d)) != NULL) {

            // Check to make sure this is a pcd file match
            if (std::strlen(dir->d_name) >= 4 &&
                    std::strcmp(dir->d_name + std::strlen(dir->d_name)  - 4, ".pcd") == 0) {
                pcd_fns.push_back(dir_name + "/" + std::string(dir->d_name));
            }
        }

        closedir(d);
    }

    return pcd_fns;
}

int
main (int argc, char** argv)
{
    std::string strDirectory;
    std::string strGTFile;
    if ( !pcl::console::parse_argument(argc, argv, "-s", strDirectory) ) {
        std::cout << "Need source directory (-s)\n";
        return (-1);
    }
    if ( !pcl::console::parse_argument(argc, argv, "-t", strGTFile) ) {
        std::cout << "Need ground truth file (-t)\n";
        return (-1);
    }

    std::vector<std::string> pcd_fns = get_pcd_in_dir(strDirectory);
    std::sort(pcd_fns.begin(),pcd_fns.end());

    std::cout << "PCD FILES\n";
    for(std::string s: pcd_fns) {
        std::cout << s << std::endl;
    }

    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y,%H-%M-%S");
    auto dateStr = oss.str();

    std::ofstream foutSICP;
    foutSICP.open(dateStr+"SICPkitti.csv");

    std::ofstream foutGICP;
    foutGICP.open(dateStr+"GICPkitti.csv");

    std::ofstream foutse3GICP;
    foutse3GICP.open(dateStr+"se3GICPkitti.csv");

    KittiMetrics semanticICPMetrics(strGTFile, &foutSICP);
    KittiMetrics se3GICPMetrics(strGTFile, &foutse3GICP);
    KittiMetrics GICPMetrics(strGTFile, &foutGICP);

    for(size_t n = 0; n<100; n++) {
        std::cout << "Cloud# " << n << std::endl;
        std::string strTarget = pcd_fns[n];
        std::string strSource = pcd_fns[n+3];
        //Sophus::SE3d initTransform = semanticICPMetrics.getGTtransfrom(n, n+3);
        Eigen::Matrix4d temp = Eigen::Matrix4d::Identity();
        Sophus::SE3d initTransform(temp);
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudA (new pcl::PointCloud<pcl::PointXYZL>);

        if (pcl::io::loadPCDFile<pcl::PointXYZL> (strSource, *cloudA) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read source file\n");
            return (-1);
        }

        std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
            semanticA (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

        semanticicp::pcl_2_semantic(cloudA, semanticA);
        semanticA->removeSemanticClass( 3 );
        semanticA->removeSemanticClass( 10 );
        semanticA->removeSemanticClass( 11 );
        cloudA = semanticA->getpclPointCloud();

        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudB (new pcl::PointCloud<pcl::PointXYZL>);

        if (pcl::io::loadPCDFile<pcl::PointXYZL> (strTarget, *cloudB) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read target file\n");
            return (-1);
        }

        std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
            semanticB (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

        semanticicp::pcl_2_semantic(cloudB, semanticB);
        semanticB->removeSemanticClass( 3 );
        semanticB->removeSemanticClass( 10 );
        semanticB->removeSemanticClass( 11 );
        cloudB = semanticB->getpclPointCloud();


        semanticicp::SemanticIterativeClosestPoint<pcl::PointXYZ, uint32_t> sicp;
        sicp.setInputSource(semanticA);
        sicp.setInputTarget(semanticB);

        auto begin = std::chrono::steady_clock::now();
        sicp.align(semanticA, initTransform);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Time Multiclass: "
                << std::chrono::duration_cast<std::chrono::seconds>(end-begin).count() << std::endl;
        Sophus::SE3d sicpTranform = sicp.getFinalTransFormation();
        std::cout << "SICP MSE: "
                  << semanticICPMetrics.evaluate(sicpTranform, n, n+3)
                  << std::endl;

        for(size_t t = 0; t< cloudA->points.size(); t++){
            pcl::PointXYZL p = cloudA->points[t];
            p.label=0;
            cloudA->points[t] = p;
        }
        std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
            semanticAnoL (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

        semanticicp::pcl_2_semantic(cloudA, semanticAnoL);


        for(size_t t = 0; t< cloudB->points.size(); t++){
            pcl::PointXYZL p = cloudB->points[t];
            p.label=0;
            cloudB->points[t] = p;
        }
        std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
            semanticBnoL (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

        semanticicp::pcl_2_semantic(cloudB, semanticBnoL);

        semanticicp::SemanticIterativeClosestPoint<pcl::PointXYZ, uint32_t> sicp2;
        sicp2.setInputSource(semanticAnoL);
        sicp2.setInputTarget(semanticBnoL);

        begin = std::chrono::steady_clock::now();
        sicp2.align(semanticAnoL, initTransform);
        end = std::chrono::steady_clock::now();
        std::cout << "Time Single Class: "
                  << std::chrono::duration_cast<std::chrono::seconds>(end-begin).count() << std::endl;
        Sophus::SE3d gicpTransform = sicp2.getFinalTransFormation();
        std::cout << "se3GICP MSE: "
                  << se3GICPMetrics.evaluate(gicpTransform, n, n+3)
                  << std::endl;

        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZL, pcl::PointXYZL> gicp;
        gicp.setInputCloud(cloudA);
        gicp.setInputTarget(cloudB);
        pcl::PointCloud<pcl::PointXYZL> final1;

        begin = std::chrono::steady_clock::now();
        gicp.align(final1, (initTransform.matrix()).cast<float>());
        end = std::chrono::steady_clock::now();
        std::cout << "Time GICP: "
                  << std::chrono::duration_cast<std::chrono::seconds>(end-begin).count() << std::endl;
        Eigen::Matrix4f mat = gicp.getFinalTransformation();
        std::cout << "Final GICP Transform\n";
        std::cout << mat << std::endl;
        Sophus::SE3d gicpTransform2(mat.cast<double>());
        std::cout << "GICP MSE: "
                  << GICPMetrics.evaluate(gicpTransform2, n, n+3)
                  << std::endl;

    }
    std::cout << " SICP FINAL MSE: " << semanticICPMetrics.getTransformMSE() << std::endl;
    std::cout << "Transform\n";
    semanticICPMetrics.printTransfrom();
    std::cout << "Rot\n";
    semanticICPMetrics.printRot();
    std::cout << "Trans\n";
    semanticICPMetrics.printTrans();

    std::cout << " se3GICP FINAL MSE: " << se3GICPMetrics.getTransformMSE() << std::endl;
    std::cout << "Transform\n";
    se3GICPMetrics.printTransfrom();
    std::cout << "Rot\n";
    se3GICPMetrics.printRot();
    std::cout << "Trans\n";
    se3GICPMetrics.printTrans();

    std::cout << " GICP FINAL MSE: " << GICPMetrics.getTransformMSE() << std::endl;
    std::cout << "Transform\n";
    GICPMetrics.printTransfrom();
    std::cout << "Rot\n";
    GICPMetrics.printRot();
    std::cout << "Trans\n";
    GICPMetrics.printTrans();

    foutSICP.close();
    foutse3GICP.close();
    foutGICP.close();

    return (0);
}
