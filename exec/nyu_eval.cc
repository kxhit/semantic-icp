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
#include "nyu_metrics.h"
#include "bootstrap.h"


struct NotDigit {
    bool operator()(const char c) {
        return !std::isdigit(c);
    }
};

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
    std::string strTestFile;
    if ( !pcl::console::parse_argument(argc, argv, "-s", strDirectory) ) {
        std::cout << "Need source directory (-s)\n";
        return (-1);
    }
    if ( !pcl::console::parse_argument(argc, argv, "-t", strTestFile) ) {
        std::cout << "Need file (-t)\n";
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

/*
    std::ofstream foutSICP;
    foutSICP.open(dateStr+"SICPnyu.csv");

    std::ofstream foutGICP;
    foutGICP.open(dateStr+"GICPnyu.csv");

    std::ofstream foutse3GICP;
    foutse3GICP.open(dateStr+"se3GICPnyu.csv");
*/
    NYUMetrics semanticICPMetrics(strTestFile, dateStr+"SICPnyu.csv", 0);
    NYUMetrics se3GICPMetrics(strTestFile, dateStr+"se3GICPnyu.csv", 0);
    NYUMetrics GICPMetrics(strTestFile, dateStr+"GICPnyu.csv", 0);

    while(semanticICPMetrics.morePairs()) {
        std::vector<size_t> pairs = semanticICPMetrics.getPairs();
        for(size_t n=0; n< (pairs.size()-1); n++) {
        size_t indxS = pairs[n];
        size_t indxT = pairs[n+1];
        std::string strTarget = pcd_fns[indxT];
        std::string strSource = pcd_fns[indxS];
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudA (new pcl::PointCloud<pcl::PointXYZL>);
        std::cout << "Source Cloud " << strSource << std::endl;
        std::string numSource = strSource;
        NotDigit nd;
        std::remove_if(numSource.begin(), numSource.end(), nd);
        int num = std::stoi(numSource);
        std::cout << "cloud number " << num << std::endl;

        if (pcl::io::loadPCDFile<pcl::PointXYZL> (strSource, *cloudA) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read source file\n");
            return (-1);
        }

        std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
            semanticA (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

        semanticicp::pcl_2_semantic(cloudA, semanticA);
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudB (new pcl::PointCloud<pcl::PointXYZL>);

        if (pcl::io::loadPCDFile<pcl::PointXYZL> (strTarget, *cloudB) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read target file\n");
            return (-1);
        }

        std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
            semanticB (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

        semanticicp::pcl_2_semantic(cloudB, semanticB);

        //Sophus::SE3d initTransform = semanticICPMetrics.getGTtransfrom(n, n+3);
        Eigen::Matrix4d temp = Eigen::Matrix4d::Identity();
        //Bootstrap boot(cloudA, cloudB);
        //Eigen::Matrix4d temp = (boot.align()).cast<double>();
        Sophus::SE3d initTransform(temp);

        semanticicp::SemanticIterativeClosestPoint<pcl::PointXYZ, uint32_t> sicp;
        sicp.setInputSource(semanticA);
        sicp.setInputTarget(semanticB);

        auto begin = std::chrono::steady_clock::now();
        sicp.align(semanticA, initTransform);
        auto end = std::chrono::steady_clock::now();
        std::cout << "Time Multiclass: "
                << std::chrono::duration_cast<std::chrono::seconds>(end-begin).count() << std::endl;
        Sophus::SE3d sicpTranform = sicp.getFinalTransFormation();
        std::cout << "Final SICP Transform\n";
        std::cout << sicpTranform.matrix() << std::endl;

        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudASICP (new pcl::PointCloud<pcl::PointXYZL>());
        pcl::transformPointCloud(*cloudA, *cloudASICP, (sicpTranform.matrix()).cast<float>());
        std::cout << "SICP Accuracy "
                  << semanticICPMetrics.evaluate(cloudASICP, cloudB, numSource)
                  << std::endl;

        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudAnoL (new pcl::PointCloud<pcl::PointXYZL>());
        for(size_t t = 0; t< cloudA->points.size(); t++){
            pcl::PointXYZL p = cloudA->points[t];
            p.label=0;
            cloudAnoL->push_back(p);
        }
        std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
            semanticAnoL (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

        semanticicp::pcl_2_semantic(cloudAnoL, semanticAnoL);


        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudBnoL (new pcl::PointCloud<pcl::PointXYZL>());
        for(size_t t = 0; t< cloudB->points.size(); t++){
            pcl::PointXYZL p = cloudB->points[t];
            p.label=0;
            cloudBnoL->push_back(p);
        }
        std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
            semanticBnoL (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

        semanticicp::pcl_2_semantic(cloudBnoL, semanticBnoL);

        semanticicp::SemanticIterativeClosestPoint<pcl::PointXYZ, uint32_t> sicp2;
        sicp2.setInputSource(semanticAnoL);
        sicp2.setInputTarget(semanticBnoL);

        begin = std::chrono::steady_clock::now();
        sicp2.align(semanticAnoL, initTransform);
        end = std::chrono::steady_clock::now();
        std::cout << "Time Single Class: "
                  << std::chrono::duration_cast<std::chrono::seconds>(end-begin).count() << std::endl;
        Sophus::SE3d gicpTransform = sicp2.getFinalTransFormation();
        std::cout << "Final se3GICP Transform\n";
        std::cout << gicpTransform.matrix() << std::endl;

        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudAse3GICP (new pcl::PointCloud<pcl::PointXYZL>());
        pcl::transformPointCloud(*cloudA, *cloudAse3GICP, (gicpTransform.matrix()).cast<float>());
        std::cout << "se3GICP Accuracy "
                  << se3GICPMetrics.evaluate(cloudAse3GICP, cloudB, numSource)
                  << std::endl;


        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZL, pcl::PointXYZL> gicp;
        gicp.setInputCloud(cloudA);
        gicp.setInputTarget(cloudB);

        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudAGICP (new pcl::PointCloud<pcl::PointXYZL>());

        begin = std::chrono::steady_clock::now();
        gicp.align(*cloudAGICP, (initTransform.matrix()).cast<float>());
        end = std::chrono::steady_clock::now();
        std::cout << "Time GICP: "
                  << std::chrono::duration_cast<std::chrono::seconds>(end-begin).count() << std::endl;
        Eigen::Matrix4f mat = gicp.getFinalTransformation();
        std::cout << "Final GICP Transform\n";
        std::cout << mat << std::endl;
        std::cout << "GICP Accuracy "
                  << GICPMetrics.evaluate(cloudAGICP, cloudB, numSource)
                  << std::endl;
        }

    }
    return (0);
}
