//
//  main.c
//  MonteCarlo_v1
//
//  Created by Cheryl Shang on 2015-06-03.
//  Copyright (c) 2015 Cheryl Shang. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/cl.hpp>
#include <time.h>


#define MAX_SOURCE_SIZE (0x100000)

int main( int argc, char* argv[] )
{
    clock_t tic = clock();
    
    int S_init = 100;
    int N_sim = 10000;
    double T = 1.00;
    double mu = 0.1;
    
    size_t globalSize, localSize, groupSize;
    localSize = 64;
    globalSize = ceil(N_sim/(double)localSize)*localSize;
    groupSize = ceil(N_sim/(double)localSize);
    
    double *S_old = (double*)malloc(sizeof(double)*N_sim);
    double *S_new = (double*)malloc(sizeof(double)*N_sim);
    
    
    for (int i = 0; i < N_sim; i++) {
        S_old[i] = S_init;
        //S_new[i] = S_init;
    }
    
    
    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;
    
    fp = fopen("mc.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char*)malloc(MAX_SOURCE_SIZE);
    source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose( fp );

    
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    
    cl_int err;
    
    
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
    
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
    
    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    
    // Create a command queue
    queue = clCreateCommandQueue(context, device_id, 0, &err);
    
    
    program = clCreateProgramWithSource(context, 1,
                                        (const char **) & source_str, NULL, &err);
    //printf("%s\n", source_str);
    // Build the program executable
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    //clBuildProgram(program, 1, NULL, "-I \\Users\\cherylshang\\Downloads\\Random123-1.08\\include//Random123", NULL, NULL);
    
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "mc", &err);
    //kernel2 = clCreateKernel(program, "reduce", &err);
    
    // Create the input and output arrays in device memory for our calculation
    cl_mem s_old_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                          N_sim * sizeof(double), NULL, NULL);
    cl_mem s_new_mem_obj = clCreateBuffer(context, CL_MEM_READ_WRITE,
                                          N_sim * sizeof(double), NULL, NULL);

    
    err = clEnqueueWriteBuffer(queue, s_old_mem_obj, CL_TRUE, 0, N_sim * sizeof(double), S_old, 0, NULL, NULL);
    
    err = clEnqueueWriteBuffer(queue, s_new_mem_obj, CL_TRUE, 0, N_sim * sizeof(double), S_new, 0, NULL, NULL);
    
    // Set the arguments to our compute kernel
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &s_old_mem_obj);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &s_new_mem_obj);

    
    
    // Execute the kernel over the entire range of the data set
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                 0, NULL, NULL);
    
    clFinish(queue);
    
    
    // Read the results from the device
    clEnqueueReadBuffer(queue, s_new_mem_obj, CL_TRUE, 0,
                        N_sim * sizeof(double), S_new, 0, NULL, NULL);
    // Display the result to the screen
    
    double sum = 0;
    double temp = 0;
    for(int i = 0; i < N_sim; i++){
        //printf("%f\n", S_new[i]);
        sum += S_new[i];
        temp += S_new[i];
        if((i+1)%64 == 0){
            printf("%f\n", temp);
            temp=0;
        }
        //if(S_new[i] <1) count++;
    }
    //printf("%f", count * 1.0 / N_sim);
    printf("%f\n", sum / N_sim * exp(-mu*T));
    
    // release OpenCL resources
    clReleaseMemObject(s_old_mem_obj);
    clReleaseMemObject(s_new_mem_obj);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    
    //release host memory
    free(S_old);
    free(S_new);
    
    clock_t toc = clock();
    
    printf("Elapsed: %f seconds\n", (double)(toc - tic) / CLOCKS_PER_SEC);
    return 0;
}

