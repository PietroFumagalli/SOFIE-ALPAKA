#ifndef TOPK_KERNEL_HPP
#define TOPK_KERNEL_HPP

#include <alpaka/alpaka.hpp>

namespace alpaka_kernels {

// MaxK is the capacity of the register buffer:
// if k <= MaxK, use fast register memory,
// if k > MaxK, fallback to slower global memory
template <int MaxK = 64>
struct TopKKernel {
    template <typename TAcc, typename T, typename Dim, typename Idx>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, T const* input, T* output,
                                  alpaka::Vec<Dim, Idx> input_strides,
                                  alpaka::Vec<Dim, Idx> output_strides,
                                  alpaka::Vec<Dim, Idx> output_shape,
                                  Idx topk_axis, Idx topk_axis_size,
                                  Idx k) const {
        using DimAcc = alpaka::Dim<TAcc>;
        static_assert(DimAcc::value == Dim::value,
                      "Accelerator and data dimensions must match!");

        if (k == 0) return;

        constexpr std::size_t D = Dim::value;
        auto elements = alpaka::uniformElementsND(acc, output_shape);

        for (auto const& idx : elements) {
            Idx input_idx = 0;
            Idx output_idx = 0;

            for (std::size_t d = 0; d < D; ++d) {
                Idx const coord = idx[d];
                input_idx += coord * input_strides[d];
                output_idx += coord * output_strides[d];
            }

            Idx const input_topk_axis_stride = input_strides[topk_axis];
            Idx const output_topk_axis_stride = output_strides[topk_axis];

            if constexpr (MaxK > 0) {
                if (k <= MaxK) {
                    // Use registers (faster)
                    T top_vals[MaxK];
                    Idx count = 0;

                    for (Idx j = 0; j < topk_axis_size; ++j) {
                        Idx const curr_idx =
                            input_idx + (j * input_axis_stride);
                        T const val = input[curr_idx];

                        if (count == k && val <= cache_vals[k - 1]) continue;

                        Idx insert_pos = 0;
                        while (insert_pos < count &&
                               val <= cache_vals[insert_pos]) {
                            insert_pos++;
                        }

                        if (insert_pos < k) {
                            Idx const end_shift = (count < k) ? count : k - 1;
                            for (Idx s = end_shift; s > insert_pos; --s) {
                                top_vals[s] = top_vals[s - 1];
                            }

                            top_vals[insert_pos] = val;
                            if (count < k) count++;
                        }
                    }

                    // Write to output
                    for (Idx i = 0; i < k; ++i) {
                        Idx const write_idx =
                            output_idx + (i * output_axis_stride);
                        if (i < count)
                            output[write_idx] = top_vals[i];
                        else
                            output[write_idx] = static_cast<T>(0);
                    }

                    // Return early so we don't run the slow path
                    continue;
                }
            }

            // Use global memory (slower)
            Idx count = 0;

            for (Idx j = 0; j < topk_axis_size; ++j) {
                Idx const curr_idx = input_idx + (j * input_axis_stride);
                T const val = input[curr_idx];

                if (count == k) {
                    Idx const last_pos_idx =
                        output_idx + ((k - 1) * output_axis_stride);
                    if (val <= output[last_pos_idx]) continue;
                }

                Idx insert_pos = 0;
                while (insert_pos < count) {
                    Idx const curr_out_idx =
                        output_idx + (insert_pos * output_axis_stride);
                    if (val > output[curr_out_idx]) break;
                    insert_pos++;
                }

                if (insert_pos < k) {
                    Idx const end_shift = (count < k) ? count : k - 1;

                    // Shift in global memory, this is the slow part
                    for (Idx s = end_shift; s > insert_pos; --s) {
                        Idx const src =
                            output_idx + ((s - 1) * output_axis_stride);
                        Idx const dst = output_idx + (s * output_axis_stride);
                        output[dst] = output[src];
                    }

                    Idx const insert_dst =
                        output_idx + (insert_pos * output_axis_stride);
                    output[insert_dst] = val;

                    if (count < k) count++;
                }
            }

            for (Idx i = count; i < k; ++i) {
                Idx const write_idx = output_idx + (i * output_axis_stride);
                output[write_idx] = static_cast<T>(0);
            }
        }
    }
};

}  // namespace alpaka_kernels

#endif  // TOPK_KERNEL_HPP
