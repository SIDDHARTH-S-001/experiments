// Header guard: It prevents the file from being included multiple times accidentally when you compile.
#ifndef INCLUDE_AKFSFSIM_KALMANFILTER_H
#define INCLUDE_AKFSFSIM_KALMANFILTER_H

#include <vector> // brings vector container from standard c++ lib.
#include <Eigen/Dense> // brings c++ lib for linear algebra.

#include "car.h"
#include "sensors.h"
#include "beacons.h"

using Eigen::VectorXd; // dynamic size vector
using Eigen::Vector2d; // 2D vector of doubles
using Eigen::Vector4d; // 4D vector

using Eigen::MatrixXd; // dynamic size matrix
using Eigen::Matrix2d; // 2x2 matrix
using Eigen::Matrix4d; // 4x4 matrix

class KalmanFilterBase // defining a base class (contains basic functionality)
{
    public: // things inside can be used outside the class.

        KalmanFilterBase():m_initialised(false){} // constructor: when object is created, it gets initialized to false.
        virtual ~KalmanFilterBase(){} // destructor, called when object is destroyed. its virtual because later we have inheritance.
        void reset(){m_initialised = false;} // set filter back to uninitialized.
        bool isInitialised() const {return m_initialised;}

    protected: // only the base class and its child classes can access these contents.
    
        VectorXd getState() const {return m_state;} // returns current state
        MatrixXd getCovariance()const {return m_covariance;} // Return the current covariance matrix (uncertainty in the state).
        void setState(const VectorXd& state ) {m_state = state; m_initialised = true;} // Set the internal state to a new vector and mark the filter as initialized.
        void setCovariance(const MatrixXd& cov ){m_covariance = cov;} // Set the internal covariance matrix.

    private: // only accessed by the base class.
        bool m_initialised; // whether the filter has been initialized or not.
        VectorXd m_state; // current estimated state
        MatrixXd m_covariance; // uncertainity in current state
};

class KalmanFilter : public KalmanFilterBase // new Kalman filter class that inherits from KalmanFilterBase. It reuses the functionality already defined.
{
    public:

        VehicleState getVehicleState(); // Returns the current best estimate of the vehicle's state.
        Matrix2d getVehicleStatePositionCovariance(); // Returns how much uncertainty there is about the vehicle's position.

        void predictionStep(double dt); // Predict the next state just based on time (dt = delta time).
        void predictionStep(GyroMeasurement gyro, double dt); // Predict using gyro sensor + time.
        void handleLidarMeasurements(const std::vector<LidarMeasurement>& meas, const BeaconMap& map);
        void handleLidarMeasurement(LidarMeasurement meas, const BeaconMap& map);
        void handleGPSMeasurement(GPSMeasurement meas);

};

#endif  // INCLUDE_AKFSFSIM_KALMANFILTER_H