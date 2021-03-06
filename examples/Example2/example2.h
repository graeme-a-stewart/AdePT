// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLE2_H
#define EXAMPLE2_H

#include <VecGeom/base/Config.h>
#include <VecGeom/volumes/PlacedVolume.h>

#ifdef VECGEOM_ENABLE_CUDA
#include <VecGeom/management/CudaManager.h> // forward declares vecgeom::cxx::VPlacedVolume
#endif

void example2(const vecgeom::cxx::VPlacedVolume *world);

#endif
