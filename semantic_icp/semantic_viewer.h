#ifndef SEMANTIC_VIEWER_H_
#define SEMANTIC_VIEWER_H_

#include <thread>
#include <string>
#include <mutex>

#include <semantic_point_cloud.h>

namespace semanticicp
{
    struct rgb_t{
        int r;
        int g;
        int b;
    };

    template <typename PointT, typename SemanticT>
    class SemanticViewer
    {

        public:
        SemanticViewer() :
        stopped(false),
        visualizationTread(&SemanticViewer<PointT,SemanticT>::runVisualizer, this)
        {};

        ~SemanticViewer() {
            visualizationTread.join();
        };

        void addSemanticPointCloudSingleColor(
                const typename SemanticPointCloud<PointT, SemanticT>::Ptr sCloud,
                int r, int g, int b, const std::string &id );

        bool wasStopped();

        private:
        void runVisualizer();

        bool stopped;
        std::thread visualizationTread;
        std::mutex stoppedGuard;
        std::mutex cloudsGuard;
        std::vector<std::string> idsCurrent;
        std::vector<std::string> idsToAdd;
        std::map<std::string, std::pair<typename pcl::PointCloud<PointT>::Ptr, rgb_t>> cloudsToAdd;
    };

} // namespace semanticicp

#include <impl/semantic_viewer.hpp>

#endif //SEMANTIC_VIEWER_H_
