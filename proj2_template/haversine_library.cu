#include <sstream>
#include <iostream>
#include <cuda_runtime.h>


__device__ double deg2rad(double deg) {
  return deg * ((2 * acos(0.0))/180);
}

__global__ void haversine_distance_kernel(int size, const double *x1,const double *y1,
    const double *x2,const double *y2, double *dist)
{
  //use any references to compute haversine distance bewtween (x1,y1) and (x2,y2), given in vectors/arrays
  //e.g., https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
  int i = blockIdx.x * blockIdx.x + threadIdx.x;
  if (i < size){
    double lon1 = x1[i];
    double lon2 = x2[i];
    double lat1 = y1[i];
    double lat2 = y2[i];

    double R = 6371; // Radius of the earth in km
    double dLat = deg2rad(lat2-lat1);  // deg2rad below
    double dLon = deg2rad(lon2-lon1); 
    double a = 
      sin(dLat/2) * sin(dLat/2) +
      cos(deg2rad(lat1)) * cos(deg2rad(lat2)) * 
      sin(dLon/2) * sin(dLon/2)
      ; 
    double c = 2 * atan2(sqrt(a), sqrt(1-a)); 
    dist[i] = R * c; // Distance in km
  }
}



void run_kernel(int size, const double *x1,const double *y1, const double *x2,const double *y2, double *dist)
   
{
  dim3 dimBlock(1024);
  printf("in run_kernel dimBlock.x=%d\n",dimBlock.x);

  dim3 dimGrid(ceil((double)size / dimBlock.x));
  
  haversine_distance_kernel<<<dimGrid, dimBlock>>>
    (size,x1,y1,x2,y2,dist);

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::stringstream strstr;
    strstr << "run_kernel launch failed" << std::endl;
    strstr << "dimBlock: " << dimBlock.x << ", " << dimBlock.y << std::endl;
    strstr << "dimGrid: " << dimGrid.x << ", " << dimGrid.y << std::endl;
    strstr << cudaGetErrorString(error);
    throw strstr.str();
  }
}
