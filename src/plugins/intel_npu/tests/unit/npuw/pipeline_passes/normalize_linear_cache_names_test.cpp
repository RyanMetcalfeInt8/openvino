// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include "llm_compiled_model_utils.hpp"
#include "openvino/opsets/opset13.hpp"
#include "openvino/pass/make_stateful.hpp"
#include "openvino/pass/stateful_to_stateless.hpp"

namespace {
namespace ops = ov::opset13;

std::shared_ptr<ov::Model> make_restored_model(bool paired = true, bool collision = false) {
    const std::string id = "past_key_values.0.conv_statepresent.0.conv_state";
    auto p = std::make_shared<ops::Parameter>(ov::element::f32, ov::Shape{1, 8, 3});
    p->output(0).set_names({"input_restored." + id});
    auto r = std::make_shared<ops::Result>(std::make_shared<ops::Relu>(p));
    r->output(0).set_names({paired ? "output_restored." + id : "unrelated"});
    ov::ParameterVector params{p};
    ov::ResultVector results{r};
    if (collision) {
        auto other = std::make_shared<ops::Parameter>(ov::element::f32, ov::Shape{1});
        other->output(0).set_names({"cache_params.past.conv.0"});
        params.push_back(other);
        results.push_back(std::make_shared<ops::Result>(other));
    }
    return std::make_shared<ov::Model>(results, params);
}

TEST(NormalizeLinearCacheNames, StatefulRoundTrip) {
    auto beam = std::make_shared<ops::Parameter>(ov::element::i32, ov::Shape{1});
    beam->output(0).set_names({"beam_idx"});
    ov::ParameterVector params{beam};
    ov::ResultVector results;
    std::map<std::string, std::string> pairs;
    for (const auto& role : {"conv_state", "recurrent_state", "key", "value"}) {
        for (int layer : {0, 3}) {
            const auto suffix = std::to_string(layer) + "." + role;
            const auto past = "past_key_values." + suffix;
            const auto present = "present." + suffix;
            auto p = std::make_shared<ops::Parameter>(ov::element::f32, ov::Shape{1, 8, 3});
            p->output(0).set_names({past});
            auto gather = std::make_shared<ops::Gather>(p, beam, ops::Constant::create(ov::element::i64, {}, {0}));
            auto r = std::make_shared<ops::Result>(std::make_shared<ops::Relu>(gather));
            r->output(0).set_names({present});
            params.push_back(p);
            results.push_back(r);
            pairs.emplace(past, present);
        }
    }
    auto model = std::make_shared<ov::Model>(results, params);
    ov::pass::MakeStateful(pairs).run_on_model(model);
    ov::pass::StatefulToStateless().run_on_model(model);
    EXPECT_TRUE(ov::npuw::util::normalize_restored_linear_cache_names(model));
    for (int layer : {0, 3}) {
        for (const auto& role : {"conv", "ssm"}) {
            auto name = "cache_params.past." + std::string(role) + "." + std::to_string(layer);
            EXPECT_EQ(model->input(name).get_node()->get_friendly_name(), name);
            EXPECT_NO_THROW(model->output("cache_params.present." + std::string(role) + "." + std::to_string(layer)));
        }
        EXPECT_NO_THROW(model->input("past_key_values." + std::to_string(layer) + ".key"));
        EXPECT_NO_THROW(model->output("present." + std::to_string(layer) + ".value"));
    }
    EXPECT_FALSE(ov::npuw::util::normalize_restored_linear_cache_names(model));
    model->validate_nodes_and_infer_types();
}

TEST(NormalizeLinearCacheNames, MissingPairDoesNotMutate) {
    auto model = make_restored_model(false);
    const auto names = model->input().get_names();
    EXPECT_ANY_THROW(ov::npuw::util::normalize_restored_linear_cache_names(model));
    EXPECT_EQ(model->input().get_names(), names);
}

TEST(NormalizeLinearCacheNames, CollisionDoesNotMutate) {
    auto model = make_restored_model(true, true);
    const auto names = model->inputs()[0].get_names();
    EXPECT_ANY_THROW(ov::npuw::util::normalize_restored_linear_cache_names(model));
    EXPECT_EQ(model->inputs()[0].get_names(), names);
}

TEST(NormalizeLinearCacheNames, UnrelatedAndMismatchedNamesStayIntact) {
    auto model = make_restored_model();
    for (const auto& name : {"cache_params.past.conv.0",
                             "unrelated",
                             "input_restored.past_key_values.0.conv_statepresent.1.conv_state",
                             "input_restored.past_key_values.0.conv_statepresent.0.recurrent_state"}) {
        model->get_parameters()[0]->output(0).set_names({name});
        EXPECT_FALSE(ov::npuw::util::normalize_restored_linear_cache_names(model));
        EXPECT_EQ(model->input().get_any_name(), name);
    }
}
}  // namespace
