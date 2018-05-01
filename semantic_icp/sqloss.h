#ifndef __SQ_LOSS_H__
#define __SQ_LOSS_H__

#include<ceres/loss_function.h>

#include<cmath>
#include<math.h>

namespace semanticicp {

class SQLoss : public ceres::LossFunction {
  public:
    void Evaluate(double s, double rho[3]) const {
     double v = s + std::numeric_limits<double>::epsilon();
     rho[0] = std::sqrt(v);
     rho[1] = 1.0/(2.0*std::sqrt(v));
     rho[2] = -1.0/(4.0*std::pow(v,1.5));
    }
};

}
#endif // _SQ_LOSS_H_
