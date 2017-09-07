#ifndef _NYU_METRICS_H_
#define _NYU_METRICS_H_

#include "csv.h"

class NYUMetrics
{
    public:
        typedef pcl::PointXYZL PointT;

        typedef pcl::PointCloud<PointT> PointCloud;
        typedef typename PointCloud::Ptr PointCloudPtr;

        typedef pcl::KdTreeFLANN<PointT> KdTree;
        typedef typename KdTree::Ptr KdTreePtr;

        NYUMetrics(std::string& testFileName, std::string outName, size_t countStart = 0, size_t numClasses = 895) :
            testFile_(testFileName),
            numClasses_(numClasses),
            outName_(outName),
            count_(countStart)
        {
            for(CSVIterator loop(testFile_); loop != CSVIterator(); ++loop) {
                std::vector<size_t> data;
                for(size_t n =0; n < (*loop).size(); n++) {
                    data.push_back(std::stoi((*loop)[n]));
                }
                testPairs_.push_back(data);
            }
            confusion_ = Eigen::MatrixXi::Zero(numClasses_,numClasses_);
            it_ = testPairs_.begin() + countStart;
        };

        double
        evaluate(PointCloudPtr source, PointCloudPtr target, std::string label) {
            int num = std::stoi(label);
            std::ostringstream oss;
            oss << "Label" << num << "-";
            count_++;
            std::ofstream out;
            std::cout << oss.str()+outName_;
            out.open(oss.str()+outName_);

            double inlier = 0;
            double total = 0;
            KdTreePtr tree(new KdTree());
            tree->setInputCloud(target);

            for(PointT p: *source) {
                std::vector<int> nn_index; nn_index.reserve (1);
                std::vector<float> nn_dist_sq; nn_dist_sq.reserve (1);

                tree->nearestKSearch(p, 1, nn_index, nn_dist_sq);

                if ( nn_dist_sq[0]<25.0 ) {
                    uint32_t labelSource = p.label;
                    uint32_t labelTarget = target->at(nn_index[0]).label;
                    confusion_(labelSource,labelTarget)++;

                    out << labelSource << ", " << labelTarget << std::endl;
                    total++;
                    if(labelSource==labelTarget)
                        inlier++;
                }
            }
            out.close();

            std::ofstream matrixOut;
            matrixOut.open(outName_);
            matrixOut << confusion_;
            matrixOut.close();
            
            return inlier/total;
        };

        std::vector<size_t> getPairs() {
            std::vector<size_t> out;
            if( it_ != testPairs_.end()) {
                out = *it_;
                it_++;
            };
            return out;
        };

        bool morePairs() {
            return it_ != testPairs_.end();
        };

        Eigen::MatrixXi getConfusionMatrix() {
            return confusion_;
        };


    private:
        std::ifstream testFile_;
        std::vector<std::vector<size_t>> testPairs_;
        std::vector<std::vector<size_t>>::iterator it_;

        Eigen::MatrixXi confusion_;
        size_t numClasses_;

        size_t count_;

        std::string outName_;
};



#endif // _KITTI_METRICS_H_
