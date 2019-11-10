#pragma once

#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct DabnnEliminateDropout final : public PredicateBasedPass {
  explicit DabnnEliminateDropout()
      : PredicateBasedPass(
            PassType::Nop,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}

  std::string getPassName() const override {
    return "dabnn_eliminate_dropout";
  }

  bool patternMatchPredicate(Node* node) override {
    return node->kind() == kDropout;
  }

  bool runTransform(Node* node, Graph&, NodeDestroyType& destroy_current)
      override {
    // Don't assume that theres only one output.
    for (size_t i = 0; i < node->outputs().size(); ++i) {
      node->outputs()[i]->replaceAllUsesWith(node->input());
    }
    destroy_current = NodeDestroyType::DestroyOne;
    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE

