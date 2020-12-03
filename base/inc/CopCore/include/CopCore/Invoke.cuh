// SPDX-FileCopyrightText: 2020 CERN, for the benefit of the LHCb collaboration
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "backend/BackendCommon.h"
#include "Logger.h"

/**
 * @brief      Invokes a function specified by its function and arguments.
 *
 * @param[in]  function            The function.
 * @param[in]  grid_dim            Number of blocks of kernel invocation.
 * @param[in]  block_dim           Number of threads of kernel invocation.
 * @param[in]  shared_memory_size  Shared memory size.
 * @param      stream              The stream where the function will be run.
 * @param[in]  arguments           The arguments of the function.
 * @param[in]  I                   Index sequence
 *
 * @return     Return value of the function.
 */
#if defined(TARGET_DEVICE_CPU) || (defined(TARGET_DEVICE_HIP) && (defined(__HCC__) || defined(__HIP__))) || \
    ((defined(TARGET_DEVICE_CUDA) && defined(__CUDACC__)) || (defined(TARGET_DEVICE_CUDACLANG) && defined(__CUDA__)))
template <class Fn, class Tuple, unsigned long... I>
void invoke_impl(Fn &&function, const dim3 &grid_dim, const dim3 &block_dim, cudaStream_t stream,
                 const Tuple &invoke_arguments, std::index_sequence<I...>)
{
  // If any grid dimension component, or any block dimension component is zero,
  // return without running.
  if (grid_dim.x == 0 || grid_dim.y == 0 || grid_dim.z == 0 || block_dim.x == 0 || block_dim.y == 0 ||
      block_dim.z == 0) {
    return;
  }

#if defined(TARGET_DEVICE_CPU)
  _unused(stream);

  gridDim = {grid_dim.x, grid_dim.y, grid_dim.z};
  for (unsigned int i = 0; i < grid_dim.x; ++i) {
    for (unsigned int j = 0; j < grid_dim.y; ++j) {
      for (unsigned int k = 0; k < grid_dim.z; ++k) {
        blockIdx = {i, j, k};
        function(std::get<I>(invoke_arguments)...);
      }
    }
  }
#elif defined(TARGET_DEVICE_HIP) && (defined(__HCC__) || defined(__HIP__))
  hipLaunchKernelGGL(function, grid_dim, block_dim, 0, stream, std::get<I>(invoke_arguments)...);
#elif (defined(TARGET_DEVICE_CUDA) && defined(__CUDACC__)) || (defined(TARGET_DEVICE_CUDACLANG) && defined(__CUDA__))
  function<<<grid_dim, block_dim, 0, stream>>>(std::get<I>(invoke_arguments)...);
#endif
}
#else
template <class Fn, class Tuple, unsigned long... I>
void invoke_impl(Fn &&, const dim3 &, const dim3 &, cudaStream_t, const Tuple &, std::index_sequence<I...>)
{
  error_cout << "Global function invoked with unexpected backend.\n";
}
#endif
