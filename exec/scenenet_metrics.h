#ifndef _SCENENET_METRICS_H_
#define _SCENENET_METRICS_H_

#include "csv.h"

class SceneNetMetrics
{
    public:

        SceneNetMetrics(std::string& gtFileName, std::ostream *out = &std::cout) :
            gtFile_(gtFileName),
            count_(0),
            transformMSE_(0),
            rotMSE_(0),
            transMSE_(0)
        {
            for(CSVIterator loop(gtFile_); loop != CSVIterator(); ++loop) {
                Eigen::Matrix4d mat = Eigen::Matrix4d::Identity();
                double data[16];
                for(size_t n =0; n < (*loop).size() && n < 16; n++) {
                    data[n] = std::stod((*loop)[n]);
                }
                mat = Eigen::Map<Eigen::Matrix<double,4,4,Eigen::RowMajor> > (data);
                Sophus::SE3d trans = Sophus::SE3d::fitToSE3(mat);
                size_t index = static_cast<size_t>(std::stod((*loop)[16]));
                gtPoses_.push_back(trans.inverse());
                std::cout << "Index: " << index << std::endl;
                std::cout << "Transfrom: " << std::endl << trans.inverse().matrix() << std::endl;
            }
            out_ = out;

        };

        double
        evaluate(const Sophus::SE3d& transform, size_t poseIDA, size_t poseIDB, int timeSeconds, int outter_iter) {
            Sophus::SE3d transformGT = getGTtransfrom(poseIDA, poseIDB);
            Sophus::SE3d transformDiff = transformGT*transform.inverse();
            double transformError = transformDiff.log().squaredNorm();
            double rotError = transformDiff.so3().log().squaredNorm();
            double transError = transformDiff.translation().squaredNorm();

            transformDiffs_.push_back(transformDiff);
            transformErrors_.push_back(transformError);
            rotErrors_.push_back(rotError);
            transErrors_.push_back(transError);

            transformMSE_ += transformError;
            rotMSE_ += rotError;
            transMSE_ += transError;
            count_++;

            *out_ << poseIDA << ", " << poseIDB << ", "
                  << transformError << ", " <<  rotError << ", "
                  << transError << ", " << timeSeconds;
            Eigen::Matrix<double,4,4,Eigen::RowMajor> temp = transformDiff.matrix();
            for( size_t i = 0, size = temp.size(); i<size; i++) {
                *out_ << ", " << *(temp.data()+i);
            };
            temp = transform.matrix();
            for( size_t i = 0, size = temp.size(); i<size; i++) {
                *out_ << ", " << *(temp.data()+i);
            };
            *out_ << ", " << outter_iter;
            *out_ << std::endl;
            std::cout << "Pose A\n";
            std::cout << gtPoses_[poseIDA].matrix();
            std::cout << std::endl;
            std::cout << "Pose B\n";
            std::cout << gtPoses_[poseIDB].matrix();
            std::cout << std::endl;
            std::cout << "GT Transform\n";
            std::cout << transformGT.matrix();
            std::cout << std::endl;
            return transformError;
        };

        Sophus::SE3d getGTtransfrom(size_t poseIDA, size_t poseIDB) {
            const Sophus::SE3d& poseA = gtPoses_[poseIDA];
            const Sophus::SE3d& poseB = gtPoses_[poseIDB];
            return poseA.inverse()*poseB;
        };

        double getTransformMSE() {
            return transformMSE_/double(count_);
        };

        double getRotMSE() {
            return rotMSE_/double(count_);
        };

        double getTransMSE() {
            return transMSE_/double(count_);
        };

        void printTransfrom() {
            for(double v: transformErrors_) {
                std::cout << v << std::endl;
            }
        };

        void printRot() {
            for(double v: rotErrors_) {
                std::cout << v << std::endl;
            }
        };

        void printTrans() {
            for(double v: transErrors_) {
                std::cout << v << std::endl;
            }
        };


    private:
        std::ifstream gtFile_;
        std::vector<Sophus::SE3d> gtPoses_;

        std::vector<Sophus::SE3d> transformDiffs_;
        std::vector<double> transformErrors_;
        std::vector<double> rotErrors_;
        std::vector<double> transErrors_;
        double transformMSE_;
        double rotMSE_;
        double transMSE_;
        size_t count_;

        std::ostream *out_;
};



#endif // _SCENENET_METRICS_H_
