#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#include "opencl2/tinymt32_jump.clh"

double generateGaussianNoise(double mu, double sigma, tinymt32j_t *tinymt) {
    
    const double epsilon = 1.17549e-38;//std::numeric_limits<double>::min();
    const double two_pi = 2.0*3.14159265358979323846;
    
    
    
    double z0, z1;
    double u1, u2;
    
    do {
        
        u1 =  tinymt32j_single01(tinymt);
        u2 =  tinymt32j_single01(tinymt);
        
    }
    
    while ( u1 <= epsilon );
    
    
    z0 = sqrt((double)-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt((double)-2.0 * log(u1)) * sin(two_pi * u2);
    
    return z0 * sigma + mu;
    
}


__kernel void mc(__global double *S_old, __global double *S_new) {
    
    
    int i = get_global_id(0);
    
    double T = 1.00;
    double sigma = 0.25;
    double mu = 0.1;
    int N = 200;
    double K = 100;
    
    double delt = T/N;
    double drift = mu * delt;
    double sqrt_delt = sqrt((double)delt);
    double sigma_sqrt_delt = sigma * sqrt_delt;
    
    
    double z0, z1;
    double generate;
    double t;
    
    tinymt32j_t tinymt;
    tinymt32j_init_jump(&tinymt, (get_global_id(0) + get_local_id(0) ));
    
    
    while (N--) {
        S_old[i] = S_old[i] + S_old[i] * (drift + sigma_sqrt_delt * generateGaussianNoise(0, 1, &tinymt));
        S_new[i] = S_old[i] > 0.0 ? S_old[i] : 0.0;
        S_old[i] = S_new[i];
    }
    
    S_new[i] = S_new[i]-K > 0.0 ? S_new[i]-K : 0.0;
    
}

