#ifndef ALPAKA_KERNELS_CONCAT_HPP
#define ALPAKA_KERNELS_CONCAT_HPP

#include <alpaka/alpaka.hpp>
#include <array>

namespace alpaka_kernels {

struct ConcatKernel {
    template <typename TAcc, typename T, typename Dim, typename Idx,
              std::size_t N>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,

        // CHANGE 1: Use std::array
        std::array<T const*, N> input_ptrs, T* output,
        std::array<alpaka::Vec<Dim, Idx>, N> input_strides,
        std::array<Idx, N> axis_sizes,

        std::size_t num_inputs, std::size_t axis,
        alpaka::Vec<Dim, Idx> output_strides,
        alpaka::Vec<Dim, Idx> output_shape) const {
        using DimAcc = alpaka::Dim<TAcc>;
        static_assert(DimAcc::value == Dim::value,
                      "Accelerator and Data dims must match");

        constexpr std::size_t D = Dim::value;

        auto elements = alpaka::uniformElementsND(acc, output_shape);

        for (auto const& idx : elements) {
            // A. Compute Output Index
            Idx out_idx = 0;
            for (std::size_t d = 0; d < D; ++d) {
                out_idx += idx[d] * output_strides[d];
            }

            // B. Find which input matrix this pixel belongs to
            Idx axis_coord = idx[axis];
            std::size_t chosen = 0;
            Idx offset = 0;

            for (std::size_t k = 0; k < N; ++k) {
                if (k >= num_inputs) break;

                Idx sz = axis_sizes[k];
                if (axis_coord < offset + sz) {
                    chosen = k;
                    break;
                }
                offset += sz;
            }

            // C. Compute Input Index
            Idx in_idx = 0;
            for (std::size_t d = 0; d < D; ++d) {
                Idx coord_out = idx[d];
                Idx coord_in = (d == axis) ? (coord_out - offset) : coord_out;

                // std::array works with [] just like C-arrays
                in_idx += coord_in * input_strides[chosen][d];
            }

            // D. Copy
            T const* src = input_ptrs[chosen];
            output[out_idx] = src[in_idx];
        }
    }
};

}  // namespace alpaka_kernels

#endif
