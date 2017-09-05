#ifndef _BOOTSTRAP_H_
#define _BOOTSTRAP_H_

#include <pcl/features/fpfh.h>
#include <pcl/features/multiscale_feature_persistence.h>
#include <pcl/features/normal_3d.h>
#include <pcl/registration/ia_ransac.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>

class Bootstrap
{
    public:
        typedef pcl::PointXYZL PointT;

        typedef pcl::PointCloud<PointT> PointCloud;
        typedef typename PointCloud::Ptr PointCloudPtr;

        typedef pcl::PointCloud<pcl::FPFHSignature33> FeatureCloud;
        typedef typename FeatureCloud::Ptr FeatureCloudPtr;

        Bootstrap(PointCloudPtr source, PointCloudPtr target) {
            PointCloudPtr tempA( new PointCloud() );
            pcl::VoxelGrid<PointT> sor;
            sor.setInputCloud (source);
            sor.setLeafSize (0.2f, 0.2f, 0.2f);
            sor.filter (*tempA);
            sourceKeypoints_ = tempA;
            std::cout << "Source size: " << sourceKeypoints_->size() << std::endl;
            sourceFeatures_ = getFeatures(sourceKeypoints_, sourceKeypoints_);
            PointCloudPtr tempB ( new PointCloud() );
            pcl::VoxelGrid<PointT> tar;
            tar.setInputCloud (target);
            tar.setLeafSize (0.2f, 0.2f, 0.2f);
            tar.filter (*tempB);
            targetKeypoints_ = tempB;
            std::cout << "Target size: " << targetKeypoints_->size() << std::endl;
            targetFeatures_ = getFeatures(targetKeypoints_, targetKeypoints_);
        };

        Eigen::Matrix4f align() {
            pcl::CorrespondencesPtr correspondences (new pcl::Correspondences());
            pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33>
                cest;
            pcl::SampleConsensusInitialAlignment<PointT, PointT, pcl::FPFHSignature33> sac_ia;
            sac_ia.setInputSource ( sourceKeypoints_ );
            sac_ia.setSourceFeatures ( sourceFeatures_ );
            sac_ia.setInputTarget ( targetKeypoints_ );
            sac_ia.setTargetFeatures ( targetFeatures_ );
            sac_ia.setMinSampleDistance (0.1f);
            sac_ia.setMaxCorrespondenceDistance (0.4f);
            sac_ia.setMaximumIterations (500);
            PointCloudPtr cloud ( new PointCloud () );
            sac_ia.align (*cloud);
            Eigen::Matrix4f transform = sac_ia.getFinalTransformation ();
            /*
            cest.determineCorrespondences ( *correspondences );
            pcl::CorrespondencesPtr corr_filtered (new pcl::Correspondences());
            pcl::registration::CorrespondenceRejectorSampleConsensus<PointT> rejector;
            rejector.setInputSource (sourceKeypoints_);
            rejector.setInputTarget (targetKeypoints_);
            rejector.setInlierThreshold (2.5);
            rejector.setMaximumIterations (1000);
            rejector.setRefineModel (false);
            rejector.setInputCorrespondences (correspondences);;
            rejector.getCorrespondences (*corr_filtered);
            pcl::registration::TransformationEstimationSVD<PointT, PointT> trans_est;
            Eigen::Matrix4f transform;
            trans_est.estimateRigidTransformation (*sourceKeypoints_,
                    *targetKeypoints_, *corr_filtered, transform);
                    */
            std::cout << "Boostrap Alignment:\n";
            std::cout << transform << std::endl;
            return transform;

        };

    private:
        FeatureCloudPtr sourceFeatures_;
        PointCloudPtr sourceKeypoints_;
        FeatureCloudPtr targetFeatures_;
        PointCloudPtr targetKeypoints_;

        FeatureCloudPtr
        getFeatures(PointCloudPtr cloud, PointCloudPtr cloudKeypoints) {

            pcl::NormalEstimation<PointT, pcl::Normal> ne;
            ne.setInputCloud (cloud);
            pcl::search::KdTree<PointT>::Ptr tree (new
                pcl::search::KdTree<PointT> ());
            ne.setSearchMethod (tree);
            pcl::PointCloud<pcl::Normal>::Ptr normals (new
                pcl::PointCloud<pcl::Normal> ());
            ne.setRadiusSearch (3.0);
            ne.compute (*normals);

            pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33>::Ptr
                fest (new pcl::FPFHEstimation<PointT, pcl::Normal, pcl::FPFHSignature33>());
            fest->setInputCloud (cloud);
            fest->setInputNormals (normals);
            fest->setSearchMethod (tree);
            fest->setRadiusSearch(3.0);
            FeatureCloudPtr features (new FeatureCloud());
            fest->compute(*features);
            /*
            pcl::MultiscaleFeaturePersistence<PointT, pcl::FPFHSignature33>
                fper;
            boost::shared_ptr<std::vector<int> > keypoints (new std::vector<int> ());
            std::vector<float> scale_values = { 0.5f, 1.0f, 1.5f };
            fper.setScalesVector (scale_values);
            fper.setAlpha (1.3f);
            fper.setFeatureEstimator (fest);
            fper.setDistanceMetric (pcl::CS);
            FeatureCloudPtr features (new FeatureCloud());
            fper.determinePersistentFeatures (*features, keypoints);
            for(int i: *keypoints) {
                cloudKeypoints->push_back(cloud->at(i));
            };
            */
            return features;
        };
};


#endif // _BOOSTRAP_H_
