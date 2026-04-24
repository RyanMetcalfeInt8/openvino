// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/pass/convert_model.hpp"

#include "openvino/core/model.hpp"
#include "openvino/pass/manager.hpp"
#include "openvino/util/log.hpp"

// Include internal transformation headers
#include "transformations/common_optimizations/moc_transformations.hpp"
#include "transformations/flush_fp32_subnormals_to_zero.hpp"
#include "transformations/smart_reshape/smart_reshape.hpp"

namespace ov {
namespace pass {

ConvertModel::ConvertModel(const ConvertModelOptions& options) : m_options(options) {}

bool ConvertModel::run_on_model(const std::shared_ptr<ov::Model>& model) {
    if (m_options.verbose) {
        OPENVINO_INFO("Applying MOC transformations to model: ", model->get_friendly_name());
        OPENVINO_INFO("  Operations before: ", model->get_ops().size());
    }

    ov::pass::Manager manager;

    // Apply transformations in the same order as Python's convert_model()
    if (m_options.smart_reshape) {
        manager.register_pass<ov::pass::SmartReshape>();
        if (m_options.verbose) {
            OPENVINO_INFO("  Registered: SmartReshape");
        }
    }

    // Main MOC transformations pass (60+ optimizations including SDPA fusion)
    manager.register_pass<ov::pass::MOCTransformations>(m_options.constant_folding);
    if (m_options.verbose) {
        OPENVINO_INFO("  Registered: MOCTransformations (cf=", m_options.constant_folding, ")");
    }

    if (m_options.flush_fp32_subnormals) {
        manager.register_pass<ov::pass::FlushFP32SubnormalsToZero>();
        if (m_options.verbose) {
            OPENVINO_INFO("  Registered: FlushFP32SubnormalsToZero");
        }
    }

    manager.run_passes(model);

    if (m_options.verbose) {
        OPENVINO_INFO("  Operations after: ", model->get_ops().size());
        OPENVINO_INFO("MOC transformations complete");
    }

    return true;
}

}  // namespace pass
}  // namespace ov
