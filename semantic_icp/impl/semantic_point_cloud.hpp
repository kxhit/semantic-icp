#ifndef SEMANTIC_POINT_CLOUD_HPP_
#define SEMANTIC_POINT_CLOUD_HPP_

#include <cmath>
#include <iostream>

namespace semanticicp
{

    template<typename PointT, typename SemanticT>
    void SemanticPointCloud<PointT, SemanticT>::addSemanticCloud(
            SemanticT label, PointCloudPtr cloud_ptr,
            bool computeKd, bool computeCov) {

        semanticLabels.push_back(label);
        labeledPointClouds[label] = cloud_ptr;

        if( computeKd == true) {
            KdTreePtr tree(new KdTree());
            tree->setInputCloud(cloud_ptr);
            labeledKdTrees[label] = tree;

            if( computeCov == true) {
                Eigen::Vector3d mean;
                std::vector<int> nn_indecies; nn_indecies.reserve (k_correspondences_);
                std::vector<float> nn_dist_sq; nn_dist_sq.reserve (k_correspondences_);

                typename pcl::PointCloud<PointT>::const_iterator points_iterator =
                    cloud_ptr->begin ();
                MatricesVectorPtr cloud_covariances(new MatricesVector);

                for(;points_iterator != cloud_ptr->end (); ++points_iterator) {
                    const PointT &query_point = *points_iterator;
                    Eigen::Matrix3d cov;

                    cov.setZero();
                    mean.setZero();

                    tree->nearestKSearch(query_point, k_correspondences_, nn_indecies, nn_dist_sq);

                    for(int index: nn_indecies) {
                        const PointT &pt = (*cloud_ptr)[index];

                        mean[0] += pt.x;
                        mean[1] += pt.y;
                        mean[2] += pt.z;

                        cov(0,0) += pt.x*pt.x;

                        cov(1,0) += pt.y*pt.x;
                        cov(1,1) += pt.y*pt.y;

                        cov(2,0) += pt.z*pt.x;
                        cov(2,1) += pt.z*pt.y;
                        cov(2,2) += pt.z*pt.z;
                    }

                    mean /= static_cast<double> (k_correspondences_);
                    for (int k = 0; k < 3; k++){
                        for (int l = 0; l <= k; l++)
                        {
                            cov(k,l) /= static_cast<double> (k_correspondences_);
                            cov(k,l) -= mean[k]*mean[l];
                            cov(l,k) = cov(k,l);
                        }
                    }

                    Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov, Eigen::ComputeFullU);
                    cov.setZero();
                    Eigen::Matrix3d U = svd.matrixU();

                    for(int k = 0; k<3; k++) {
                        Eigen::Vector3d col = U.col(k);
                        double v = 1.;
                        if(k == 2)
                            v = epsilon_;
                        cov+= v*col*col.transpose();
                    }
                    cloud_covariances->push_back(cov);
                }
                labeledCovariances[label] = cloud_covariances;
            }
        }

    };

} // namespace semanticicp

#endif //SEMANTIC_POINT_COULD_HPP_
