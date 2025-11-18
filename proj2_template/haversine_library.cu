#include <sstream>
#include <iostream>
#include <cuda_runtime.h>
#include <cmath>
#define PI 3.14159265358979323846

__global__ void haversine_distance_kernel(int size, const double *x1,const double *y1,
    const double *x2,const double *y2, double *dist)
{
 //use any references to compute haversine distance bewtween (x1,y1) and (x2,y2), given in vectors/arrays
 //e.g., https://stackoverflow.com/questions/27928/calculate-distance-between-two-latitude-longitude-points-haversine-formula
// lat = x lon = y
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    // lat = x, lon = y
    double lat1 = y1[idx];
    double lon1 = x1[idx];
    double lat2 = y2[idx];
    double lon2 = x2[idx];

    double radius = 6371.0; // km

    // Convert to radians
    double lat1r = lat1 * PI / 180.0;
    double lat2r = lat2 * PI / 180.0;
    double dlat  = (lat2 - lat1) * PI / 180.0;
    double dlon  = (lon2 - lon1) * PI / 180.0;

    // Haversine formula
    double a = sin(dlat * 0.5) * sin(dlat * 0.5) +
               cos(lat1r) * cos(lat2r) *
               sin(dlon * 0.5) * sin(dlon * 0.5);

    double c = 2.0 * asin(sqrt(a));
    dist[idx] = radius * c;
// function getDistanceFromLatLonInKm(lat1,lon1,lat2,lon2) {
//   int R = 6371;
//   double dLat = deg2rad(x2-lat1); 
//   double dLon = deg2rad(lon2-lon1); 
//   double a = 
//     Math.sin(dLat/2) * Math.sin(dLat/2) +
//     Math.cos(deg2rad(lat1)) * Math.cos(deg2rad(lat2)) * 
//     Math.sin(dLon/2) * Math.sin(dLon/2)
//     ; 
//   double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a)); 
//   double d = R * c; // Distance in km
//   return d;
// }

// function deg2rad(deg) {
//   return deg * (Math.PI/180)
// }

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
