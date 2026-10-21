// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include "npuw_transformations/insert_vocab_sub128.hpp"
#include "openvino/opsets/opset13.hpp"

namespace {
using namespace ov;
namespace ops = ov::opset13;

TEST(NormalizeVocabWeights, FoldsOnlyTerminalQuantizationParametersAndTranspose) {
    auto hidden = std::make_shared<ops::Parameter>(element::f32, Shape{1, 1, 4});
    auto weights = ops::Constant::create(element::u8, Shape{8, 4}, {128});
    auto scale = ops::Constant::create(element::f16, Shape{8}, {0.5});
    auto zp = ops::Constant::create(element::u8, Shape{8}, {128});
    auto shape = ops::Constant::create(element::i64, Shape{2}, {8, 1});
    auto scale_reshape = std::make_shared<ops::Reshape>(scale, shape, false);
    auto zp_convert = std::make_shared<ops::Convert>(zp, element::f16);
    auto zp_reshape = std::make_shared<ops::Reshape>(zp_convert, shape, false);
    auto sub = std::make_shared<ops::Subtract>(std::make_shared<ops::Convert>(weights, element::f16),
                                             zp_reshape);
    auto mul = std::make_shared<ops::Multiply>(sub, scale_reshape);
    auto convert = std::make_shared<ops::Convert>(mul, element::f32);
    auto transpose = std::make_shared<ops::Transpose>(convert,
                                                     ops::Constant::create(element::i64, Shape{2}, {1, 0}));
    auto mm = std::make_shared<ops::MatMul>(hidden, transpose, false, false);
    mm->set_friendly_name("lm_head");
    mm->output(0).get_tensor().set_names({"logits"});
    auto model = std::make_shared<Model>(OutputVector{mm}, ParameterVector{hidden});

    ov::npuw::normalize_vocab_weights(model);
    model->validate_nodes_and_infer_types();
    auto normalized = ov::as_type_ptr<ops::MatMul>(model->get_results()[0]->get_input_node_shared_ptr(0));
    ASSERT_TRUE(normalized);
    EXPECT_TRUE(normalized->get_transpose_b());
    EXPECT_EQ(normalized->input_value(1).get_element_type(), element::f32);
    EXPECT_EQ(normalized->get_friendly_name(), "lm_head");
    EXPECT_EQ(normalized->output(0).get_names(), std::unordered_set<std::string>{"logits"});
    auto normalized_mul = normalized->get_input_node_shared_ptr(1)->get_input_node_shared_ptr(0);
    auto normalized_sub = normalized_mul->get_input_node_shared_ptr(0);
    EXPECT_TRUE(ov::is_type<ops::Constant>(normalized_mul->get_input_node_shared_ptr(1)));
    EXPECT_TRUE(ov::is_type<ops::Constant>(normalized_sub->get_input_node_shared_ptr(1)->get_input_node_shared_ptr(0)));
    EXPECT_EQ(normalized_sub->get_input_node_shared_ptr(0)->get_input_node_shared_ptr(0), weights);
    ov::npuw::normalize_vocab_weights(model);
    EXPECT_EQ(model->get_results()[0]->get_input_node_shared_ptr(0), normalized);
}

TEST(NormalizeVocabWeights, LeavesFloatingPointHeadUnchanged) {
    auto hidden = std::make_shared<ops::Parameter>(element::f32, Shape{1, 1, 4});
    auto weights = ops::Constant::create(element::f32, Shape{8, 4}, {1});
    auto transpose = std::make_shared<ops::Transpose>(weights,
                                                     ops::Constant::create(element::i64, Shape{2}, {1, 0}));
    auto mm = std::make_shared<ops::MatMul>(hidden, transpose, false, false);
    auto model = std::make_shared<Model>(OutputVector{mm}, ParameterVector{hidden});
    ov::npuw::normalize_vocab_weights(model);
    EXPECT_EQ(model->get_results()[0]->get_input_node_shared_ptr(0), mm);
}
}  // namespace
