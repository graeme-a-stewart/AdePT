// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0
#include "CopCore/CopCore.h"
#include "CopCore/Invoke.cuh"
#include <tuple>

#include <iostream>

// Mandelbrot set kernel
__global__ void mandel(unsigned x_size, unsigned y_size, int *z)
{
  // Calculate the starting position on the (x,y) grid for this thread
  unsigned x_loc = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned y_loc = blockIdx.y * blockDim.y + threadIdx.y;

  // This is the stride to march over the plane
  unsigned x_stride = blockDim.x * gridDim.x;
  unsigned y_stride = blockDim.y * gridDim.y;

  // Uncomment for some insight into how each kernel is running
  // printf("(%u, %u) of (%u, %u) - (%u, %u)\n", x_loc, y_loc, x_size, y_size, x_stride, y_stride);

  // Double loop over the x,y space by stride values to get the 2D block-stride loop...
  for (unsigned y = y_loc; y < y_size; y += y_stride) {
    for (unsigned x = x_loc; x < x_size; x += x_stride) {
      // Map the coordinates to the location in the results array
      unsigned idx = x + y * x_size;

      // Starting floating point(x,y) values
      float c_x = -2.0f + float(x) / x_size * 4.0;
      float c_y = -2.0f + float(y) / y_size * 4.0;

      // Imlement the usual Mandelbrot iterations, f_n+1 = f_n^2 + c
      unsigned it = 0;
      float m_x = c_x;
      float m_y = c_y;
      float tmp_x, m2;
      do {
        tmp_x = m_x * m_x - m_y * m_y + c_x;
        m_y = 2.0f * m_x * m_y + c_y;
        m_x = tmp_x;
        m2 = m_x * m_x + m_y * m_y;
        ++it;
      } while (it < 100 && m2 < 4.0f);
      if (it == 100)
        z[idx] = 0; // In
      else
        z[idx] = it; // Out (the number of iterations to move out is frequently used to colour code)
    }
  }
}

int main()
{
  // Define parameters for grid
  int x_grid = 1 << 12; // 4096
  int y_grid = 1 << 12;
  int grid_size = x_grid * y_grid;
  int *z = nullptr;   // Memory pointer on host
  int *d_z = nullptr; // Memory pointer on device

  // Initialize
  z = (int *)malloc(grid_size * sizeof(int));
  cudaMalloc((void **)&d_z, grid_size * sizeof(int));

  // Invoke
  //
  // The choice of the number of blocks and the number of threads
  // per-block here is not critical, but should be "resonable".
  // The CUDA runtime will map this in its usual way to
  // warps, with each block of threads jumping to different parts
  // of (x,y) space and processing a differernt part of the
  // plane.
  //
  // Here we use 8x8 blocks of 8x8 threads
  dim3 nBlocks(8,8);
  dim3 nThreads(8,8);

  auto args           = std::make_tuple(x_grid, y_grid, d_z);
  constexpr auto size = std::tuple_size_v<decltype(args)>;
  using indices       = std::make_index_sequence<size>;

  invoke_impl(mandel, nBlocks, nThreads, cudaStream_t{}, args, indices{});

  // Return values
  cudaMemcpy(z, d_z, grid_size * sizeof(int), cudaMemcpyDeviceToHost);

  // Check - print out the pattern in a 128x128 grid
  for (int y = 0; y < y_grid; y += x_grid/128) {
    for (int x = 0; x < x_grid; x += y_grid/128) {
      unsigned idx = x + y * x_grid;
      if (z[idx] > 0) {
        std::cout << ".";
      } else {
        std::cout << " ";
      }
    }
    std::cout << std::endl;
  }

  // Tidy
  cudaFree(d_z);
  free(z);
  return 0;
}