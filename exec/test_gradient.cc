#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <cmath>
#include <string>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/registration/gicp.h>
#include <pcl/console/parse.h>

#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <sophus/types.hpp>
#include <sophus/common.hpp>
#include <ceres/ceres.h>
#include <pcl/point_cloud.h>
#include <pcl/common/transforms.h>

//#include <gicp_cost_functor_autodiff.h>
#include <gicp_cost_function.h>
#include <local_parameterization_se3.h>

#include<Eigen/StdVector>
#include<ceres/gradient_checker.h>


int
main (int argc, char** argv)
{
    pcl::PointXYZ sourcePoint(7.96094,-5.25134,24.2516);
    //pcl::PointXYZ targetPoint(7.73844,-5.16017,24.3069);
    pcl::PointXYZ targetPoint(17.73844,-5.16017,14.3069);

    Eigen::Matrix3d sourceCov;
    sourceCov << 0.674143,  0.460412,  0.085842,
                 0.460412,  0.349471, -0.121288,
                 0.085842, -0.121288,  0.977386;

    Eigen::Matrix3d targetCov;
    /*
    targetCov << 0.674143,  0.460412,  0.085842,
                 0.460412,  0.349471, -0.121288,
                 0.085842, -0.121288,  0.977386;
                 */

    targetCov << 0.074143,  0.460412,  0.085842,
                 0.460412,  0.349471, -0.121288,
                 0.085842, -0.121288,  0.977386;

    std::default_random_engine engine;

    for(int i = 0; i<10; i++) {

      Sophus::SE3d estTransform, baseTransformation_;
      estTransform = Sophus::SE3d::sampleUniform(engine);

      ceres::CostFunction* cost_function = new semanticicp::GICPCostFunction(sourcePoint,
                                                                             targetPoint,
                                                                             sourceCov,
                                                                             targetCov,
                                                                             baseTransformation_);
      ceres::NumericDiffOptions numeric_diff_options;
      //numeric_diff_options.relative_step_size = 1e-13;

      std::vector<const ceres::LocalParameterization*> lp;
      lp.push_back(new semanticicp::LocalParameterizationSE3);

      ceres::GradientChecker gradient_checker(cost_function,
                                      &lp,
                                      numeric_diff_options);

      ceres::GradientChecker::ProbeResults results;
      std::vector<double *> params;
      params.push_back(estTransform.data());
      if (!gradient_checker.Probe(params.data(), 5e-16, &results)) {
             std::cout << "An error has occurred:\n";
              std::cout << results.error_log;
              std::cout << results.jacobians[0] << std::endl;
              std::cout << results.numeric_jacobians[0] << std::endl;
              std::cout << estTransform.matrix() << std::endl;
              std::cout << sourcePoint << std::endl;
              std::cout << targetPoint << std::endl;
              std::cout << sourceCov << std::endl;
              std::cout << targetCov << std::endl;

      }
    }

    return(0);
}
