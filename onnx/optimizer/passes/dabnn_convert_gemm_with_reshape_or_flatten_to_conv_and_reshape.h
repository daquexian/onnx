// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <numeric>

#include "onnx/common/assertions.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct DabnnGemmToConv final : public PredicateBasedPass {
  explicit DabnnGemmToConv()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "dabnn_convert_gemm_with_reshape_or_flatten_to_conv_and_reshape";
  }
  bool patternMatchPredicate(Node* node) override {
    if (node->kind() == kGemm) {
      const auto& input = node->inputs()[0]->node();

      if (input->kind() == kReshape || input->kind() == Symbol("Flatten")) {
        return true;
      }
    }
    return false;
  }
  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    destroy_current = NodeDestroyType::DestroyZero;
    Node* gemm = n;
    {
      const auto alpha = gemm->f(kalpha);
      const auto beta = gemm->f(kbeta);
      const auto transA = gemm->i(ktransA);
      const auto transB = gemm->i(ktransB);
      if (!(alpha == 1 && beta == 1 && transA == 0 && transB == 1)) {
        return false;
      }
    }
    auto reshape_or_flatten = gemm->inputs()[0]->node();
    if (reshape_or_flatten->output()->uses().size() != 1) {
        return false;
    }
    auto end_iter = graph.initializers().end();
    if (reshape_or_flatten->kind() == kReshape) {
      auto shape = reshape_or_flatten->inputs()[1];
      auto shape_iter = graph.getInitializer(shape->uniqueName());
      if (shape_iter == end_iter) {
        return false;
      }
    }
    if (reshape_or_flatten->kind() == Symbol("Flatten")) {
      if (reshape_or_flatten->hasAttribute(kaxis)) {
        const auto axis = reshape_or_flatten->i(kaxis);
        if (axis != 1) {
          return false;
        }
      }
    }
    const auto weight = gemm->input(1);
    const auto weight_iter = graph.getInitializer(weight->uniqueName());
    if (weight_iter == end_iter) {
      return false;
    }
    if (gemm->inputs().size() == 3) {
      const auto bias = gemm->input(2);
      const auto bias_iter = graph.getInitializer(bias->uniqueName());
      if (bias_iter == end_iter) {
        return false;
      }
    }
    auto conv = graph.create(kConv, gemm->inputs(), gemm->outputs().size());

    Tensor W = *weight_iter;
    W.sizes().push_back(1);
    W.sizes().push_back(1);
    Value* new_W_value = graph.addInitializerAndInput(W);
    Value* old_W_value = conv->inputs()[1];
    conv->replaceInput(1, new_W_value);
    if (old_W_value->uses().size() == 0) {
      graph.eraseInitializerAndInput(old_W_value);
    }

    conv->insertBefore(reshape_or_flatten);
    gemm->replaceAllUsesWith(conv);
    reshape_or_flatten->replaceAllUsesWith(reshape_or_flatten->input(0)->node());
    for (int i = 0; i < static_cast<int64_t>(conv->outputs().size()); ++i) {
      conv->outputs()[i]->copyMetadata(gemm->outputs()[i]);
    }

    destroy_current = NodeDestroyType::DestroyTwo;

    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE
