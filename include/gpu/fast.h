#ifndef __FAST_H__
#define __FAST_H__

#include <vector>
#include "opencv2/opencv.hpp"

//#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cuda_runtime.h>
//#include <cuda/Cuda.hpp>
//#include <helper_cuda.h>

typedef struct FastCorner
{
  int set;
  int value;
}Corner;

class GpuFast 
{
  int m_rows;
  int m_cols;
  unsigned char* m_pData;
  Corner* m_pCorner;

  cudaStream_t m_stream;

public:
  GpuFast();
  ~GpuFast();

  void create(int width,int height);

  void detect(cv::Mat image, std::vector<cv::KeyPoint>& keyPoints, int threshold=20,bool nonmaxSuppression=true);

  void destroy();

};


__device__
extern int position(int m,int n,int width);

__global__
extern void fast(uchar* image, int width, int height,Corner* d_corner,int gridsize_x, int gridsize_y, const int threshold);

__global__
extern void nms(uchar* image, Corner* d_corner,int width, int height);

#endif
