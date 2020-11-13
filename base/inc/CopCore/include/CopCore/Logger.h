// SPDX-FileCopyrightText: 2020 CERN, for the benefit of the LHCb collaboration
// SPDX-License-Identifier: Apache-2.0

#ifndef COPCORE_LOGGER
#define COPCORE_LOGGER

#define verbose_cout logger::logger(logger::verbose)
#define debug_cout logger::logger(logger::debug)
#define info_cout logger::logger(logger::info)
#define warning_cout logger::logger(logger::warning)
#define error_cout logger::logger(logger::error)

#include <iosfwd>
#include <ostream>
#include <streambuf>
#include <memory>
#include "LoggerCommon.h"

namespace logger {
class Logger {
public:
  int verbosityLevel = 3;
};

std::ostream &logger(int requestedLogLevel);

int verbosity();

void setVerbosity(int level);
} // namespace logger

#endif  // COPCORE_LOGGER
