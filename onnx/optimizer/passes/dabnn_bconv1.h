// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

// Before:
//   Z = Conv(X, Y)
//   B = Z + A
// After:
//   B = Conv(X, Y, A)
//
// the pass can handle the following cases:
//   case 1: A is 1D tensor and A.dim[0] == Z.dim[1]
//   case 2: A is 1-element 1D tensor

#include <numeric>

#include "onnx/common/assertions.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct DabnnBconv1 final : public PredicateBasedPass {
  explicit DabnnBconv1()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "dabnn_bconv1_strict";
  }
  bool patternMatchPredicate(Node* node) override {
    if (node->kind() == kConv) {
      const auto &input0 = node->inputs()[0]->node();
      const auto &second_order_input0 = input0->input()->node();
      bool branch1_match = (input0->kind() == kPad && second_order_input0->kind() == Symbol("Sign")) ||
        (input0->kind() == Symbol("Sign") && second_order_input0->kind() == kPad);
      if (branch1_match) {
        const auto &input1 = node->inputs()[1]->node();
        if (input1->kind() == Symbol("Sign")) {
          return true;
        }
      }
    }
    return false;
  }
  bool runTransform(Node* n, Graph& /*graph*/, NodeDestroyType& destroy_current)
      override {
    destroy_current = NodeDestroyType::DestroyZero;
    Node *conv = n;
    auto orig_acti = conv->inputs()[0];
    auto second_order_orig_acti = orig_acti->node()->inputs()[0];
    auto orig_weight = conv->inputs()[1];
    // check if the three ops are only used by binary conv
    if (orig_acti->uses().size() > 1 && second_order_orig_acti->uses().size() > 1 && orig_weight->uses().size() > 1) {
      return false;
    }
    // check if Conv doesn't have not its own pads
    if (conv->hasAttribute(kpads)) {
      const auto &conv_pads = conv->is(kpads);
      if (std::any_of(conv_pads.begin(), conv_pads.end(),
          [](int64_t local_value) { return local_value != 0; })) {
        return false;
      }
    }
    
    Node *sign, *pad;
    if (orig_acti->node()->kind() == kPad) {
      pad = orig_acti->node();
      sign = second_order_orig_acti->node();
    } else {
      pad = second_order_orig_acti->node();
      sign = orig_acti->node();
    }
    
    std::string pad_mode;
    if (pad->hasAttribute(kmode)) {
      pad_mode = pad->s(kmode);
    } else {
      pad_mode = "constant";
    }
    float value = 0.0;
    if (pad->hasAttribute(kvalue)) {
      value = static_cast<float>(pad->f(kvalue));
    }

    // check if Pad is used to pad -1 on the input
    if (pad_mode != "constant" || value != -1.0) {
      return false;
    }

    std::vector<int64_t> pads = pad->is(kpads);
    int pads_size = static_cast<int>(pads.size());

    // check if padding is applied only on feature dims
    if (pads[0] != 0 || pads[1] != 0 ||
        pads[pads_size / 2] != 0 || pads[pads_size / 2 + 1] != 0) {
      return false;
    }

    conv->setDomain("dabnn");
    int conv_pads_size = pads_size - 4;
    std::vector<int64_t> conv_pads(conv_pads_size, 0);

    for (int i = 2, j = 0; i < pads_size / 2; ++i, ++j) {
      conv_pads[j] += pads[i];
      conv_pads[conv_pads_size / 2 + j] += pads[pads_size / 2 + i];
    }

    conv->is_(kpads, std::move(conv_pads));
    conv->replaceInput(0, second_order_orig_acti->node()->inputs()[0]);
    conv->replaceInput(1, orig_weight->node()->inputs()[0]);
    sign->destroy();
    pad->destroy();
    orig_weight->node()->destroy();

    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE

