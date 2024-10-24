/* Copyright 2024 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef XLA_TOOLS_HLO_OPT_TRANSFORMS_PASS_REGISTRY_H_
#define XLA_TOOLS_HLO_OPT_TRANSFORMS_PASS_REGISTRY_H_

#include <functional>
#include <string>
#include <utility>

#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/log.h"
#include "third_party/protobuf/repeated_ptr_field.h"
#include "xla/hlo/pass/hlo_pass_pipeline.h"
#include "xla/tools/hlo_opt/transforms_example_passes.h"

namespace xla {

// Map of pass names to pass registration functions. The pass registration
// function takes a HloPassPipeline and adds the corresponding pass to it.
using PassRegistry =
    absl::flat_hash_map<std::string, std::function<void(HloPassPipeline&)>>;

template <typename T>
std::pair<std::string, std::function<void(HloPassPipeline&)>> RegisterPass() {
  return std::make_pair(std::string(T().name()),
                        [](HloPassPipeline& p) { p.AddPass<T>(); });
}

inline const PassRegistry& GetRegisteredPasses() {
  // Register HLO passes here if you want the hlo-opt tool
  // to be able to apply them.
  static const absl::NoDestructor<PassRegistry> registry(
      {RegisterPass<FooToBarModulePass>(),
       RegisterPass<BarToHelloModulePass>()});
  return *registry;
}

inline void buildTransformsPassPipeline(
    HloPassPipeline& transforms_pipeline,
    const ::proto2::RepeatedPtrField<std::string>& input_pass_names) {
  auto registry = GetRegisteredPasses();
  for (const auto& pass_name : input_pass_names) {
    auto it = registry.find(pass_name);
    if (it != registry.end()) {
      it->second(transforms_pipeline);
    } else {
      LOG(ERROR) << "Pass " << pass_name << " not found.";
    }
  }
}

}  // namespace xla

#endif  // XLA_TOOLS_HLO_OPT_TRANSFORMS_PASS_REGISTRY_H_
