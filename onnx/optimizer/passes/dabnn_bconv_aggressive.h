// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <numeric>

#include "onnx/common/assertions.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct DabnnBconvAggressive final : public PredicateBasedPass {
  explicit DabnnBconvAggressive()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "dabnn_bconv_aggressive";
  }
  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kConv;
  }
  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    destroy_current = NodeDestroyType::DestroyZero;
    Node *conv = n;

    
    // check if the weight is binary (+1/-1)
    const auto& conv_inputs = conv->inputs();
    auto W_iter = graph.getInitializer(conv_inputs[1]->uniqueName());
    auto end_iter = graph.initializers().end();
    if (W_iter == end_iter) {
      return false;
    }
    ONNX_ASSERT(W_iter->sizes().size() > 2);
    const Tensor W = *W_iter;
    if (W.elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return false;
    }
    const auto *ptr = W.data<float>();
    for (int i = 0; i < W.size_from_dim(0); i++) {
      if (*ptr != 1.f && *ptr != -1.f) {
        return false;
      }
      ptr++;
    }

    conv->setDomain("dabnn");

    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE



