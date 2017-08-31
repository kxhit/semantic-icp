#include <vector>
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <iomanip>
#include <ctime>
#include <sstream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/console/parse.h>

#include <semantic_point_cloud.h>
#include <semantic_icp.h>
#include <pcl_2_semantic.h>
#include "roc_metrics.h"



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
    /*
    if ( !pcl::console::parse_argument(argc, argv, "-t", strGTFile) ) {
        std::cout << "Need ground truth file (-t)\n";
        return (-1);
    }*/

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
    foutSICP.open(dateStr+"SICP.csv");
    ROCMetrics rocSICP(&foutSICP);

    std::ofstream foutGICP;
    foutGICP.open(dateStr+"GICP.csv");
    ROCMetrics rocGICP(&foutGICP);

    for(size_t n = 0; n<(pcd_fns.size()-1); n++) {
        std::cout << "Cloud# " << n << std::endl;
        std::string strTarget = pcd_fns[n];
        std::string strSource = pcd_fns[n+1];
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

        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudASICP = semanticA->getpclPointCloud();

        rocSICP.evaluate(cloudASICP, cloudB);

        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZL, pcl::PointXYZL> gicp;
        gicp.setInputCloud(cloudA);
        gicp.setInputTarget(cloudB);
        pcl::PointCloud<pcl::PointXYZL> final1;

        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudAGICP (new pcl::PointCloud<pcl::PointXYZL>);
        begin = std::chrono::steady_clock::now();
        gicp.align(*cloudAGICP, (initTransform.matrix()).cast<float>());
        end = std::chrono::steady_clock::now();
        std::cout << "Time GICP: "
                  << std::chrono::duration_cast<std::chrono::seconds>(end-begin).count() << std::endl;
        Eigen::Matrix4f mat = gicp.getFinalTransformation();
        std::cout << "Final GICP Transform\n";
        std::cout << mat << std::endl;
        Sophus::SE3d gicpTransform2(mat.cast<double>());

        rocGICP.evaluate(cloudAGICP, cloudB);

    }

    foutSICP.close();
    foutGICP.close();

    return (0);
}
