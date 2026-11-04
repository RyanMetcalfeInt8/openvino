// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "core/operator_set.hpp"
#include "exceptions.hpp"
#include "openvino/opsets/opset14.hpp"
#include "utils/common.hpp"

namespace ov::frontend::onnx::org_openvinotoolkit::opset_1 {
ov::OutputVector recurrent_gdn(const ov::frontend::onnx::Node& node) {
    namespace op = ov::opset14;
    const auto inputs = node.get_ov_inputs();
    CHECK_VALID_NODE(node, inputs.size() == 6, "RecurrentGDN requires six inputs");
    for (size_t i = 0; i < 6; ++i) {
        CHECK_VALID_NODE(node,
                         inputs[i].get_element_type() == ov::element::f32,
                         "RecurrentGDN v1 requires FLOAT inputs");
        const auto rank = inputs[i].get_partial_shape().rank();
        CHECK_VALID_NODE(node,
                         rank.is_static() && rank.get_length() == (i == 3 || i == 4 ? 3 : 4),
                         "RecurrentGDN input rank does not match its native-layout contract");
    }
    auto scalar = [](int64_t value) {
        return op::Constant::create(ov::element::i64, ov::Shape{}, {value});
    };
    auto axes = [](int64_t value) {
        return op::Constant::create(ov::element::i64, ov::Shape{1}, {value});
    };
    auto iteration = std::make_shared<op::Parameter>(ov::element::i64, ov::Shape{});
    ov::ParameterVector slices;
    for (size_t i = 0; i < 5; ++i) {
        auto shape = inputs[i].get_partial_shape();
        shape[2] = 1;
        slices.push_back(std::make_shared<op::Parameter>(ov::element::f32, shape));
    }
    auto state = std::make_shared<op::Parameter>(ov::element::f32, inputs[5].get_partial_shape());
    auto buffer = std::make_shared<op::Parameter>(ov::element::f32, inputs[2].get_partial_shape());
    auto q = std::make_shared<op::Squeeze>(slices[0], axes(2));
    auto k = std::make_shared<op::Squeeze>(slices[1], axes(2));
    auto v = std::make_shared<op::Squeeze>(slices[2], axes(2));
    auto decay = std::make_shared<op::Unsqueeze>(std::make_shared<op::Exp>(slices[3]), axes(-1));
    auto gated = std::make_shared<op::Multiply>(state, decay);
    auto key_column = std::make_shared<op::Unsqueeze>(k, axes(-1));
    auto retrieved =
        std::make_shared<op::ReduceSum>(std::make_shared<op::Multiply>(gated, key_column), axes(-2), false);
    auto delta = std::make_shared<op::Multiply>(std::make_shared<op::Subtract>(v, retrieved), slices[4]);
    auto updated = std::make_shared<op::Add>(
        gated,
        std::make_shared<op::Multiply>(key_column, std::make_shared<op::Unsqueeze>(delta, axes(-2))));
    auto step = std::make_shared<op::ReduceSum>(
        std::make_shared<op::Multiply>(updated, std::make_shared<op::Unsqueeze>(q, axes(-1))),
        axes(-2),
        true);
    auto written = std::make_shared<op::ScatterUpdate>(buffer,
                                                       std::make_shared<op::Unsqueeze>(iteration, axes(0)),
                                                       step,
                                                       scalar(2));
    auto condition = op::Constant::create(ov::element::boolean, ov::Shape{}, {true});
    ov::ParameterVector parameters{iteration};
    parameters.insert(parameters.end(), slices.begin(), slices.end());
    parameters.push_back(state);
    parameters.push_back(buffer);
    auto body = std::make_shared<ov::Model>(ov::OutputVector{condition, updated, written}, parameters);
    auto value_shape = std::make_shared<op::ShapeOf>(inputs[2]);
    auto trip = std::make_shared<op::Gather>(value_shape, scalar(2), scalar(0));
    auto loop = std::make_shared<op::Loop>(trip, condition);
    loop->set_function(body);
    for (size_t i = 0; i < 5; ++i) {
        loop->set_sliced_input(slices[i], inputs[i], 0, 1, 1, -1, 2);
    }
    loop->set_merged_input(state, inputs[5], updated);
    auto zero = op::Constant::create(ov::element::f32, ov::Shape{}, {0.f});
    loop->set_merged_input(buffer, std::make_shared<op::Broadcast>(zero, value_shape), written);
    loop->set_special_body_ports({0, 0});
    // The matcher needs sequence first on the Loop; ONNX callers receive state first.
    auto sequence = loop->get_iter_value(written, -1);
    auto final_state = loop->get_iter_value(updated, -1);
    return {final_state, sequence};
}

ONNX_OP("RecurrentGDN", OPSET_SINCE(1), org_openvinotoolkit::opset_1::recurrent_gdn, OPENVINO_ONNX_DOMAIN);
}  // namespace ov::frontend::onnx::org_openvinotoolkit::opset_1
