// ATTENTION: The code in this file is highly EXPERIMENTAL.
// Adventurous users should note that the APIs will probably change.

#pragma once

#include <numeric>

#include "onnx/common/assertions.h"
#include "onnx/optimizer/pass.h"

namespace ONNX_NAMESPACE {
namespace optimization {

struct DabnnBconvSoft final : public PredicateBasedPass {
  explicit DabnnBconvSoft()
      : PredicateBasedPass(
            PassType::Fuse,
            PassEfficiency::Complete,
            PassOptimizationType::Compute) {}
  std::string getPassName() const override {
    return "dabnn_bconv_soft";
  }
  bool patternMatchPredicate(Node* node) override {
    if (node->kind() == kConv) {
      const auto &input0 = node->inputs()[0]->node();
      bool branch1_match = input0->kind() == kPad;
      return branch1_match;
    }
    return false;
  }
  bool runTransform(Node* n, Graph& graph, NodeDestroyType& destroy_current)
      override {
    destroy_current = NodeDestroyType::DestroyZero;
    Node *conv = n;

    // check if the pad op is only used by binary conv
    if (conv->inputs()[0]->uses().size() > 1) {
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
    
    auto pad = conv->inputs()[0]->node();
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
    ONNX_ASSERT(!W.is_raw_data());
    for (const auto x : W.floats()) {
      if (x != 1.f && x != -1.f) {
        return false;
      }
    }

    conv->setDomain("dabnn");
    int conv_pads_size = pads_size - 4;
    std::vector<int64_t> conv_pads(conv_pads_size, 0);

    for (int i = 2, j = 0; i < pads_size / 2; ++i, ++j) {
      conv_pads[j] += pads[i];
      conv_pads[conv_pads_size / 2 + j] += pads[pads_size / 2 + i];
    }
    std::cout << "pads: ";
    for (const auto x : conv_pads) {
      std::cout << x;
    }
    std::cout << std::endl;

    conv->is_(kpads, std::move(conv_pads));
    conv->replaceInput(0, pad->inputs()[0]);
    pad->destroy();

    return true;
  }
};

} // namespace optimization
} // namespace ONNX_NAMESPACE


