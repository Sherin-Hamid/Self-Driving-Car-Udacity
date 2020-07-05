/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::normal_distribution;
//using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;
  particles.clear();
  
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  std::default_random_engine gen;
  
  
  for(int i=0; i<num_particles; i++){
    
    Particle p;
    p.id = i;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1.0;
    particles.push_back(p);
  }
  
  is_initialized = true;
  
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::default_random_engine gen;

  // generate random Gaussian noise
  std::normal_distribution<double> N_x(0, std_pos[0]);
  std::normal_distribution<double> N_y(0, std_pos[1]);
  std::normal_distribution<double> N_theta(0, std_pos[2]);
  
 // Particle p;
  
  for(int i=0; i<num_particles; i++){
    
    //p = particles[i];
    
    if(fabs(yaw_rate) < 0.0001){
      particles[i].x += velocity * delta_t * cos(yaw_rate);
      particles[i].y += velocity * delta_t * sin(yaw_rate);
      particles[i].theta += yaw_rate * delta_t;
    } 
    else{
      particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    }

    // adding noise
    particles[i].x += N_x(gen);
    particles[i].y += N_y(gen);
    particles[i].theta += N_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, 
                                     std::vector<LandmarkObs>& observations){
  /**
   * TODO: Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  
  double min_dist;
  double cur_dist;
  int map_id;
    
  for(int i=0; i<observations.size(); i++){
    
    LandmarkObs land_o = observations[i];

    //initialize minimum distance to the maximum possible
    min_dist = std::numeric_limits<double>::max();

    map_id = -1;
    
    for(int j=0; j<predicted.size(); j++){
      
      LandmarkObs land_p = predicted[j];
      
      //get distance between observed and predicted landmarks
      cur_dist = dist(land_o.x, land_o.y, land_p.x, land_p.y);

      // find the landmark nearest to the observed one
      if(cur_dist < min_dist){
        min_dist = cur_dist;
        map_id = land_p.id;
      }
    }

    // set the observation's id to the nearest landmark's id
    observations[i].id = map_id;
  }
  
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const std::vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  Particle p;
  double distance, cos_theta, sin_theta, lm_x, lm_y, temp_x, temp_y, trans_x, trans_y, pred_x, pred_y, sig_x, sig_y, obs_w;
  int lm_id, trans_id;
  
  for(int i=0; i<num_particles; i++){

    p = particles[i];
    p.weight = 1.0;

    // create a vector to hold the map landmark locations predicted to be within sensor range of the particle
    std::vector<LandmarkObs> predictions;

    // for each map landmark
    for(int j=0; j<map_landmarks.landmark_list.size();j++){
      
      lm_x = map_landmarks.landmark_list[j].x_f;
      lm_y = map_landmarks.landmark_list[j].y_f;
      lm_id = map_landmarks.landmark_list[j].id_i;
      
      distance = dist(p.x, p.y, lm_x, lm_y);
      if(distance < sensor_range){
        predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
    }

    // create a list of observations transformed from vehicle coordinates to map coordinates
    std::vector<LandmarkObs> obs_map;
    cos_theta = cos(p.theta);
    sin_theta = sin(p.theta);
    
    for(int j=0; j<observations.size(); j++){
      temp_x = cos_theta*observations[j].x - sin_theta*observations[j].y + p.x;
      temp_y = sin_theta*observations[j].x + cos_theta*observations[j].y + p.y;
      obs_map.push_back(LandmarkObs{observations[j].id, temp_x, temp_y});
    }

    dataAssociation(predictions, obs_map);

    for(int j=0; j<obs_map.size(); j++){
     
      trans_x = obs_map[j].x;
      trans_y = obs_map[j].y;
      trans_id = obs_map[j].id;

      for(int k=0; k<predictions.size(); k++){
        if(predictions[k].id == trans_id){
          pred_x = predictions[k].x;
          pred_y = predictions[k].y;
        }
      }

      // calculate weight for this observation with multivariate Gaussian
      sig_x = std_landmark[0];
      sig_y = std_landmark[1];
      obs_w = (1 /(2 * M_PI * sig_x * sig_y)) * exp(-(pow(pred_x - trans_x, 2) / (2 * pow(sig_x , 2)) + (pow(pred_y -trans_y, 2) / (2 * pow(sig_y, 2)))));

      p.weight *= obs_w;
      particles[i] = p;
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  
  std::vector<double> weights;
  weights.clear();
  
  for(int i=0; i<num_particles; i++){
    weights.push_back(particles[i].weight);
  }
  
  // create resampled particles
  std::vector<Particle> resampled_particles;
  resampled_particles.clear();
  
  std::default_random_engine gen;
  std::uniform_int_distribution<int> int_dist(0, num_particles-1);
  int index = int_dist(gen);
  
  double max_w = *max_element(weights.begin(), weights.end()); 
  
  std::uniform_real_distribution<double> real_dist(0.0, max_w);

  double beta = 0.0;
  
  // resample the particles according to weights
  for(int i=0; i<num_particles; i++){
    beta += real_dist(gen) * 2.0;
    while (beta > weights[index]){
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    resampled_particles.push_back(particles[index]);
  }

  // assign the resampled_particles to the previous particles
  particles = resampled_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, 
                                     const std::vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

std::string ParticleFilter::getAssociations(Particle best) {
  std::vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

std::string ParticleFilter::getSenseCoord(Particle best, std::string coord) {
  std::vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  std::string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}