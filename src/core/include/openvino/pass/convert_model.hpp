// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @file convert_model.hpp
 * @brief Pass for applying Model Optimizer Conversion (MOC) transformations
 *
 * This header provides public C++ API for applying the same optimizations
 * that Python's convert_model() applies, including:
 * - Attention fusion (ScaledDotProductAttention)
 * - Constant folding and algebraic simplifications
 * - Operation fusion (SoftPlus, Swish, GELU, etc.)
 * - 60+ graph optimization passes
 *
 * @note This is particularly important for models deployed on NPU, as
 * compile_model() does not apply these transformations automatically.
 */

#pragma once

#include <memory>

#include "openvino/pass/pass.hpp"

namespace ov {
namespace pass {

/**
 * @brief Options for model optimization
 */
struct ConvertModelOptions {
    /**
     * @brief Enable smart reshape transformation for dynamic shapes
     * @default true
     */
    bool smart_reshape = true;

    /**
     * @brief Use constant folding (cf parameter in Python API)
     * Setting to true may increase model size but improve performance
     * @default false
     */
    bool constant_folding = false;

    /**
     * @brief Enable FP32 subnormal values flushing to zero
     * @default true
     */
    bool flush_fp32_subnormals = true;

    /**
     * @brief Enable verbose logging of applied transformations
     * @default false
     */
    bool verbose = false;
};

/**
 * @brief Applies Model Optimizer Conversion (MOC) transformations to a model.
 *
 * This pass applies the same 60+ optimization passes that Python's
 * convert_model() applies, including:
 * - Attention fusion: Fuses decomposed attention patterns into
 *   ScaledDotProductAttention operations
 * - Constant folding: Pre-computes constant operations
 * - Operation fusion: Combines multiple operations (SoftPlus, Swish, GELU, etc.)
 * - Algebraic simplifications: Simplifies mathematical expressions
 * - Transpose/Reshape optimizations: Eliminates redundant operations
 * - And 50+ more transformations
 *
 * Can be used standalone or composed with ov::pass::Manager:
 * @code
 * ov::Core core;
 * auto model = core.read_model("model.onnx");
 *
 * // Standalone
 * ov::pass::Manager manager;
 * manager.register_pass<ov::pass::ConvertModel>();
 * manager.run_passes(model);
 *
 * // Or with custom options:
 * ov::pass::ConvertModelOptions opts;
 * opts.constant_folding = true;
 * manager.register_pass<ov::pass::ConvertModel>(opts);
 * manager.run_passes(model);
 * @endcode
 *
 * \ingroup ov_pass_cpp_api
 */
class OPENVINO_API ConvertModel : public ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ConvertModel");

    explicit ConvertModel(const ConvertModelOptions& options = ConvertModelOptions{});
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    ConvertModelOptions m_options;
};

}  // namespace pass
}  // namespace ov
