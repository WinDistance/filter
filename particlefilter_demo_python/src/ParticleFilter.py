import numpy as np
from numpy.random import uniform, randn
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data





class ParticleFilter(object):

    def __init__(self, N, map_size, landmarks):
        self.N = N
        self.landmarks = landmarks
        self.particles = []
        self.map_size=map_size

        # distribute particles randomly with uniform weight
        for _ in range(N):
            particle = dict()
            particle['x'] =  np.random.uniform(map_size[0], map_size[1])
            particle['y'] =  np.random.uniform(map_size[2], map_size[3])
            particle['theta'] = np.random.uniform(-np.pi, np.pi)
            particle['weight'] = 1./N
            self.particles.append(particle)

    def mean_pose(self):
        """
        calculate the mean pose of a particle set.
        
        for x and y, the mean position is the mean of the particle coordinates
        
        for theta, we cannot simply average the angles because of the wraparound 
        (jump from -pi to pi). Therefore, we generate unit vectors from the 
        angles and calculate the angle of their average 
        """
        # save x and y coordinates of particles
        xs = []
        ys = []

        # save unit vectors corresponding to particle orientations 
        vxs_theta = []
        vys_theta = []

        for particle in self.particles:
            xs.append(particle['x'])
            ys.append(particle['y'])

            #make unit vector from particle orientation
            vxs_theta.append(np.cos(particle['theta']))
            vys_theta.append(np.sin(particle['theta']))

        #calculate average coordinates
        mean_x = np.mean(xs)
        mean_y = np.mean(ys)
        mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

        return [mean_x, mean_y, mean_theta]


    def predict(self,odometry):
        """
        predict:Samples new particle positions, based on old positions, the odometry
        measurements and the motion noise 
        """
        print("Predicting particle positions")
        delta_rot1 = odometry['r1']
        delta_trans = odometry['t']
        delta_rot2 = odometry['r2']

        # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
        noise = [0.1, 0.1, 0.05, 0.05]

        # generate new particle set after motion update
        new_particles = []
        
        # standard deviations of motion noise
        sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
        sigma_delta_trans = noise[2] * delta_trans + \
        noise[3] * (abs(delta_rot1) + abs(delta_rot2))
        sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans
        # "move" each particle according to the odometry measurements plus sampled noise
        for particle in self.particles:
            new_particle = dict()
            #sample noisy motions
            noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
            noisy_delta_trans = delta_trans + np.random.normal(0, sigma_delta_trans)
            noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)
            #calculate new particle pose
            new_particle['x'] = particle['x'] + \
            noisy_delta_trans * np.cos(particle['theta'] + noisy_delta_rot1)
            new_particle['y'] = particle['y'] + \
            noisy_delta_trans * np.sin(particle['theta'] + noisy_delta_rot1)
            new_particle['theta'] = particle['theta'] + \
            noisy_delta_rot1 + noisy_delta_rot2
            new_particles.append(new_particle)


        self.particles=new_particles

    def update(self,sensor_data):
        """
        update:eval_sensor_model
        Computes the observation likelihood of all particles, given the
        particle and landmark positions and sensor measurements

        The employed sensor model is range only.
        """
        print("Update observation likelihood of all particles")
        sigma_r = 0.2

        #measured landmark ids and ranges
        ids = sensor_data['id']
        ranges = sensor_data['range']

        weights = []

        #rate each particle
        for particle in self.particles:
            all_meas_likelihood = 1.0 #for combining multiple measurements
            #loop for each observed landmark
            for i in range(len(ids)):
                lm_id = ids[i]
                meas_range = ranges[i]
                lx = self.landmarks[lm_id][0]
                ly = self.landmarks[lm_id][1]
                px = particle['x']
                py = particle['y']
                #calculate expected range measurement
                meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2 )
                #evaluate sensor model (probability density function of normal distribution)
                meas_likelihood = scipy.stats.norm.pdf(meas_range, meas_range_exp, sigma_r)
                #combine (independent) measurements
                all_meas_likelihood = all_meas_likelihood * meas_likelihood
                weights.append(all_meas_likelihood)
                particle['weight']=all_meas_likelihood
  


        #normalize weights
        normalizer = sum(weights)
        for particle in self.particles:
            particle['weight'] = particle['weight']/ normalizer





    def resample_particles(self):
        # Returns a new set of particles obtained by performing
        # stochastic universal sampling, according to the particle weights.
        print("resample_particles")
        new_particles = []

        # distance between pointers
        step = 1.0/len(self.particles)
        rand = np.random.random() * step
        cum_weight = self.particles[0]['weight']
        # index of weight container and corresponding particle
        i = 0
        #loop over all particle weights
        for particle in self.particles:
        #go through the weights until you find the particle
        #to which the pointer points
            cur_spot = rand + i * step
            while cur_spot > cum_weight and i<len(self.particles)-1:
                i = i + 1
                cum_weight +=particle['weight']
                #add that particle
                new_particles.append(particle)
                particle['weight']=step


        self.particles=new_particles



    def visualization(self,sensor_data):
        # Visualizes the state of the particle filter.
        #
        # Displays the particle cloud, mean position and landmarks.
        
        xs = []
        ys = []

        for particle in self.particles:
            xs.append(particle['x'])
            ys.append(particle['y'])

        # landmark positions
        lx=[]
        ly=[]

        for i in range (len(self.landmarks)):
            lx.append(self.landmarks[i+1][0])
            ly.append(self.landmarks[i+1][1])

        # mean pose as current estimate
        estimated_pose = self.mean_pose()

        # plot filter state
        plt.clf()
        plt.plot(xs, ys, 'r.')
        plt.plot(lx, ly, 'bo',markersize=10)
        plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
        plt.axis(self.map_size)

        #flush time
        plt.pause(1)

def main():
    # implementation of a particle filter for robot pose estimation

    print("Reading landmark positions")
    #add random seed for generating comparable pseudo random numbers
    np.random.seed(123)

    #plot preferences, interactive plotting mode
    #plt.axis([-1, 12, 0, 10])
    plt.ion()
    plt.show()
    #{1: [2.0, 1.0], 2: [0.0, 4.0], 3: [2.0, 7.0], 4: [9.0, 2.0], 5: [10.0, 5.0],...}
    landmarks = read_world("../data/world.dat")
   
    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")
   

    #initialize the particles
    map_limits = [-1, 12, 0, 10]
    pf=ParticleFilter(1000,map_limits,landmarks)
    #run particle filter
    for timestep in range(len(sensor_readings)//2):

        #plot the current state
        pf.visualization(sensor_readings[timestep, 'odometry'])

        #predict particles by sampling from motion model with odometry info
        pf.predict(sensor_readings[timestep,'odometry'])

        #calculate importance weights according to sensor model
        pf.update(sensor_readings[timestep, 'sensor'])

        #resample new particle set according to their importance weights
        pf.resample_particles()

if __name__ == "__main__":
    
    main()