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

#include <em_icp.h>
#include <gicp.h>
#include <pcl_2_semantic.h>
#include "read_confusion_matrix.h"
#include "scenenet_metrics.h"


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
    std::string strCMFile;
    if ( !pcl::console::parse_argument(argc, argv, "-s", strDirectory) ) {
        std::cout << "Need source directory (-s)\n";
        return (-1);
    }
    if ( !pcl::console::parse_argument(argc, argv, "-t", strGTFile) ) {
        std::cout << "Need ground truth file (-t)\n";
        return (-1);
    }
    if ( !pcl::console::parse_argument(argc, argv, "-m", strCMFile) ) {
        std::cout << "Need ground confusion matrix file (-m)\n";
        return (-1);
    }
    Eigen::Matrix<double, 13, 13> cm = ReadConfusionMatrix<13>(strCMFile);
    std::cout << "Confusion Matrix:\n" << cm << std::endl;

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
    //foutSICP.open(dateStr+"SICPkitti.csv");

    std::ofstream foutGICP;
    //foutGICP.open(dateStr+"GICPkitti.csv");

    std::ofstream foutse3GICP;
    //foutse3GICP.open(dateStr+"se3GICPkitti.csv");

    KittiMetrics semanticICPMetrics(strGTFile);
    KittiMetrics se3GICPMetrics(strGTFile);
    KittiMetrics GICPMetrics(strGTFile);
  */  

    std::ofstream foutSICP;
    foutSICP.open(dateStr+"EMICPscenenet.csv");

    std::ofstream foutGICP;
    foutGICP.open(dateStr+"GICPscenenet.csv");

    std::ofstream foutse3GICP;
    foutse3GICP.open(dateStr+"se3GICPscenenet.csv");

    std::ofstream foutBootstrap;
    foutBootstrap.open(dateStr+"initscenenet.csv");

    SceneNetMetrics semanticICPMetrics(strGTFile, &foutSICP);
    SceneNetMetrics se3GICPMetrics(strGTFile, &foutse3GICP);
    SceneNetMetrics GICPMetrics(strGTFile, &foutGICP);
    SceneNetMetrics bootstrapMetrics(strGTFile, &foutBootstrap);
    int STEP =1;

    for(size_t n = 290; n<pcd_fns.size()-STEP; n+=STEP) {
        std::string strTarget = pcd_fns[n];
        std::string strSource = pcd_fns[n+STEP];
        std::cout << "Target Cloud " << strTarget << std::endl;
        std::cout << "Source Cloud " << strSource << std::endl;
        size_t indxTarget = n;
        size_t indxSource = (n+STEP);
        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudA (new pcl::PointCloud<pcl::PointXYZL>);

        if (pcl::io::loadPCDFile<pcl::PointXYZL> (strSource, *cloudA) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read source file\n");
            return (-1);
        }


        /*
        std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
            semanticA (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

        semanticicp::pcl_2_semantic(cloudA, semanticA);
        semanticA->removeSemanticClass( 3 );
        semanticA->removeSemanticClass( 10 );
        semanticA->removeSemanticClass( 11 );
        //cloudA = semanticA->getpclPointCloud();
        */

        pcl::PointCloud<pcl::PointXYZL>::Ptr cloudB (new pcl::PointCloud<pcl::PointXYZL>);

        if (pcl::io::loadPCDFile<pcl::PointXYZL> (strTarget, *cloudB) == -1) //* load the file
        {
            PCL_ERROR ("Couldn't read target file\n");
            return (-1);
        }


        /*
        std::shared_ptr<semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t>>
            semanticB (new semanticicp::SemanticPointCloud<pcl::PointXYZ, uint32_t> ());

        semanticicp::pcl_2_semantic(cloudB, semanticB);
        semanticB->removeSemanticClass( 3 );
        semanticB->removeSemanticClass( 10 );
        semanticB->removeSemanticClass( 11 );
        //cloudB = semanticB->getpclPointCloud();
        */

        auto begin = std::chrono::steady_clock::now();
        //Sophus::SE3d initTransform = semanticICPMetrics.getGTtransfrom(n, n+3);
        Eigen::Matrix4d temp = Eigen::Matrix4d::Identity();
        //Bootstrap boot(cloudA, cloudB);
        //Eigen::Matrix4d temp = (boot.align()).cast<double>();
        Sophus::SE3d initTransform(temp);
        auto end = std::chrono::steady_clock::now();
        int timeInit = std::chrono::duration_cast<std::chrono::seconds>(end-begin).count();
        std::cout << "Init MSE "
                  << bootstrapMetrics.evaluate(initTransform, indxTarget, indxSource, timeInit, 0)
                  << std::endl;

        semanticicp::EmIterativeClosestPoint<13> emicp(20, 1e-6);
        pcl::PointCloud<pcl::PointXYZL>::Ptr
          finalCloudem( new pcl::PointCloud<pcl::PointXYZL> );

        begin = std::chrono::steady_clock::now();
        emicp.setSourceCloud(cloudA);
        emicp.setTargetCloud(cloudB);
        emicp.setConfusionMatrix(cm);
        emicp.align(finalCloudem, initTransform);
        end = std::chrono::steady_clock::now();
        int timeSICP = std::chrono::duration_cast<std::chrono::seconds>(end-begin).count();
        std::cout << "Time Multiclass: "
                << timeSICP << std::endl;
        Sophus::SE3d sicpTranform = emicp.getFinalTransFormation();
        std::cout << "SICP MSE: "
                  << semanticICPMetrics.evaluate(sicpTranform, indxTarget,
                          indxSource, timeSICP, emicp.getOuterIter())
                  << std::endl;

        pcl::PointCloud<pcl::PointXYZL>::Ptr
          labeledCloudem( new pcl::PointCloud<pcl::PointXYZL> );
        emicp.getFusedLabels(labeledCloudem, sicpTranform);
        std::ostringstream strstream;
        strstream << indxSource << ".pcd";
        pcl::io::savePCDFileASCII(strstream.str(), *labeledCloudem);

        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudAnoL (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile<pcl::PointXYZ> (strSource, *cloudAnoL);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloudBnoL (new pcl::PointCloud<pcl::PointXYZ>);
        pcl::io::loadPCDFile<pcl::PointXYZ> (strTarget, *cloudBnoL);


        semanticicp::GICP<pcl::PointXYZ> gicpse3(20, 1e-6);
        pcl::PointCloud<pcl::PointXYZ>::Ptr finalCloudse3( new pcl::PointCloud<pcl::PointXYZ> );

        begin = std::chrono::steady_clock::now();
        gicpse3.setSourceCloud(cloudAnoL);
        gicpse3.setTargetCloud(cloudBnoL);
        gicpse3.align(finalCloudse3);
        end = std::chrono::steady_clock::now();
        int timese3GICP = std::chrono::duration_cast<std::chrono::seconds>(end-begin).count();
        std::cout << "Time Single Class: "
                  << timese3GICP << std::endl;
        Sophus::SE3d gicpTransform = gicpse3.getFinalTransFormation();
        std::cout << "se3GICP MSE: "
                  << se3GICPMetrics.evaluate(gicpTransform, indxTarget,
                          indxSource, timese3GICP, gicpse3.getOuterIter())
                  << std::endl;

        pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZL, pcl::PointXYZL> gicp;
        pcl::PointCloud<pcl::PointXYZL> final1;

        begin = std::chrono::steady_clock::now();
        gicp.setInputCloud(cloudA);
        gicp.setInputTarget(cloudB);
        gicp.setMaxCorrespondenceDistance(1.5);
        gicp.setMaximumIterations(50);
        gicp.align(final1, (initTransform.matrix()).cast<float>());
        end = std::chrono::steady_clock::now();
        int timeGICP = std::chrono::duration_cast<std::chrono::seconds>(end-begin).count();
        std::cout << "Time GICP: "
                  << timeGICP << std::endl;
        Eigen::Matrix4f mat = gicp.getFinalTransformation();
        std::cout << "Final GICP Transform\n";
        std::cout << mat << std::endl;
        Sophus::SE3d gicpTransform2 = Sophus::SE3d::fitToSE3(mat.cast<double>());
        std::cout << "GICP MSE: "
                  << GICPMetrics.evaluate(gicpTransform2, indxTarget, indxSource, timeGICP, 1)
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
