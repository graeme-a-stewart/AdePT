// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef TRACKH
#define TRACKH

#include <VecGeom/base/Vector3D.h>
#include <VecGeom/navigation/NavStateIndex.h>

#include <curand_kernel.h>

#include <cfloat> // for FLT_MAX

enum TrackStatus { alive, dead };

struct track {
  curandState_t curand_state;
  int index{0};
  int pdg{0};
  double energy{10};
  vecgeom::Vector3D<double> pos;
  vecgeom::Vector3D<double> dir;
  vecgeom::NavStateIndex current_state;
  vecgeom::NavStateIndex next_state;
  int mother_index{0};
  TrackStatus status{alive};
  int current_process{0};
  float interaction_length{FLT_MAX};
  float energy_loss{0};         // primitive version of scoring
  int number_of_secondaries{0}; // primitive version of scoring

  __device__ float uniform() { return curand_uniform(&curand_state); }

  __device__ void SwapStates()
  {
    auto state          = this->current_state;
    this->current_state = this->next_state;
    this->next_state    = state;
  }
};

#endif
