#include "kalman_filter.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;

// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

float KalmanFilter::angle(float a){
  float PI = 4 * atan(1);
  while(a > PI || a < -1 * PI){
    if(a > PI){
      a -= (2 * PI);
    } else {
      a += (2 * PI);
    }
  }
  return a;
}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

VectorXd KalmanFilter::ComputeLaserMeasure() {
  return H_ * x_;
}

VectorXd KalmanFilter::ComputeRadarMeasure() {
  VectorXd z_pred = VectorXd(3);
  float rho = sqrt(pow(x_[0], 2) + pow(x_[1], 2));
  z_pred << rho,
            atan2(x_[1], x_[0]),
            (x_[0] * x_[2] + x_[1] * x_[3]) / rho;
  return z_pred;
}

void KalmanFilter::ComputeUpdate(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  //new estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = ComputeLaserMeasure();
  VectorXd y = z - z_pred;
  ComputeUpdate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  VectorXd z_pred = ComputeRadarMeasure();
  VectorXd y = z - z_pred;
  y(1) = angle(y[1]);
  ComputeUpdate(y);
}


