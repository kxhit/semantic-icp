#ifndef SEMANTIC_VIEWER_HPP_
#define SEMANTIC_VIEWER_HPP_

#include<chrono>

#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>

namespace semanticicp
{



template <typename PointT, typename SemanticT>
void SemanticViewer<PointT, SemanticT>::addSemanticPointCloudSingleColor(
        const typename SemanticPointCloud<PointT, SemanticT>::Ptr sCloud,
        int r, int g, int b, const std::string &id) {
    rgb_t rgb;
    rgb.r = r;
    rgb.g = g;
    rgb.b = b;

    pcl::PointCloud<PointT> cloud;
    for (SemanticT s: sCloud->semanticLabels) {
        cloud += *(sCloud->labeledPointClouds[s]);
    }
    typename pcl::PointCloud<PointT>::Ptr cloudPtr (new pcl::PointCloud<PointT>(cloud));

    std::lock_guard<std::mutex> lockClouds(cloudsGuard);
    idsToAdd.push_back(id);
    cloudsToAdd[id] = std::make_pair(cloudPtr, rgb);
};

template <typename PointT, typename SemanticT>
void SemanticViewer<PointT, SemanticT>::runVisualizer () {

    pcl::visualization::PCLVisualizer viewer ("Semantic Viewer");

    viewer.setBackgroundColor (0, 0, 0);
    viewer.addCoordinateSystem (1.0);
    viewer.initCameraParameters ();

    while (!viewer.wasStopped ())
    {
        viewer.spinOnce (100);
        std::this_thread::sleep_for (std::chrono::microseconds (100000));

        std::lock_guard<std::mutex> lockClouds(cloudsGuard);
        for(auto id: idsToAdd) {
            auto pair = cloudsToAdd[id];
            pcl::visualization::PointCloudColorHandlerCustom<PointT>
                single_color(pair.first, pair.second.r, pair.second.g, pair.second.b);
            viewer.addPointCloud<PointT>(pair.first, single_color, id);
            cloudsToAdd.erase(id);
        }
        idsToAdd.clear();
    }

    std::lock_guard<std::mutex> lockStopped(stoppedGuard);
    stopped = true;
};

template <typename PointT, typename SemanticT>
bool SemanticViewer<PointT,SemanticT>::wasStopped() {
    std::lock_guard<std::mutex> lockStopped(stoppedGuard);
    bool temp = stopped;
    return temp;
};

} // namespace semanticicp

#endif //SEMANTIC_VIEWER_HPP_
