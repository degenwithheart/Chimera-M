/*
 * Ternary Codec C++ Extension for CHIMERA-M
 * Fast bit packing/unpacking for {-1, 0, +1} quantization
 * 
 * Build:
 *   cd chimera_m_cpp
 *   pip install pybind11
 *   python setup.py build_ext --inplace
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace py = pybind11;

/*
 * Pack float32 weights to ternary {-1, 0, +1} then into int32 bit-packed format
 * 
 * Packing scheme: 16 ternary values per uint32
 * - Each ternary value encoded as 2 bits: 00=-1, 01=0, 10=+1 (3 unused)
 * - Little-endian: first value in LSB bits [0:1], second in [2:3], etc.
 * 
 * Args:
 *   weights: float32 numpy array of any shape
 *   scale: float scaling factor (max absolute value for normalization)
 *   stochastic: bool whether to use stochastic rounding
 *   seed: int random seed for stochastic mode
 * 
 * Returns:
 *   packed: int32 numpy array of shape (ceil(n/16),)
 */
py::array_t<int32_t> ternary_pack(
    py::array_t<float, py::array::c_style | py::array::forcecast> weights,
    float scale,
    bool stochastic = false,
    int seed = 0
) {
    if (scale < 1e-8f) {
        scale = 1e-8f;
    }
    
    // Get buffer info
    py::buffer_info weights_buf = weights.request();
    
    // Calculate total elements
    size_t n = 1;
    for (auto dim : weights_buf.shape) {
        n *= dim;
    }
    
    // Calculate packed size (16 values per int32)
    size_t packed_size = (n + 15) / 16;
    
    // Create output array
    auto packed = py::array_t<int32_t>(packed_size);
    py::buffer_info packed_buf = packed.request();
    
    // Get pointers
    const float* w = static_cast<const float*>(weights_buf.ptr);
    int32_t* p = static_cast<int32_t*>(packed_buf.ptr);
    
    // Initialize to zero
    std::memset(p, 0, packed_size * sizeof(int32_t));
    
    // Initialize RNG state for stochastic mode
    uint32_t rng_state = seed ? static_cast<uint32_t>(seed) : 0x12345678;
    auto rand_float = [&rng_state]() -> float {
        // Simple LCG
        rng_state = rng_state * 1103515245 + 12345;
        return static_cast<float>(rng_state & 0x7fffffff) / 0x7fffffff;
    };
    
    // Pack in groups of 16
    for (size_t group = 0; group < packed_size; group++) {
        int32_t packed_val = 0;
        
        for (int i = 0; i < 16; i++) {
            size_t idx = group * 16 + i;
            if (idx >= n) break;
            
            // Normalize
            float normalized = w[idx] / scale;
            
            // Quantize to ternary
            int code;
            if (stochastic) {
                float rand = rand_float();
                // Stochastic: probability proportional to distance from threshold
                if (normalized > 0.5f + (rand - 0.5f) * 0.1f) {
                    code = 2;  // +1
                } else if (normalized < -0.5f - (rand - 0.5f) * 0.1f) {
                    code = 0;  // -1
                } else {
                    code = 1;  // 0
                }
            } else {
                // Deterministic
                if (normalized > 0.5f) {
                    code = 2;  // +1
                } else if (normalized < -0.5f) {
                    code = 0;  // -1
                } else {
                    code = 1;  // 0
                }
            }
            
            // Pack into 2 bits at position i*2
            packed_val |= (code << (i * 2));
        }
        
        p[group] = packed_val;
    }
    
    return packed;
}

/*
 * Unpack int32 bit-packed ternary back to float32
 * 
 * Args:
 *   packed: int32 numpy array from ternary_pack
 *   original_shape: tuple of output dimensions
 *   scale: float scaling factor (must match pack scale)
 * 
 * Returns:
 *   weights: float32 numpy array of original_shape
 */
py::array_t<float> ternary_unpack(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> packed,
    py::tuple original_shape,
    float scale
) {
    // Calculate total elements
    size_t n = 1;
    std::vector<size_t> shape_vec;
    for (auto dim : original_shape) {
        size_t d = dim.cast<size_t>();
        shape_vec.push_back(d);
        n *= d;
    }
    
    // Get buffer info
    py::buffer_info packed_buf = packed.request();
    size_t packed_size = packed_buf.shape[0];
    
    // Create output array
    auto weights = py::array_t<float>(shape_vec);
    py::buffer_info weights_buf = weights.request();
    
    // Get pointers
    const int32_t* p = static_cast<const int32_t*>(packed_buf.ptr);
    float* w = static_cast<float*>(weights_buf.ptr);
    
    // Unpack
    for (size_t group = 0; group < packed_size; group++) {
        int32_t packed_val = p[group];
        
        for (int i = 0; i < 16; i++) {
            size_t idx = group * 16 + i;
            if (idx >= n) break;
            
            // Extract 2 bits at position i*2
            int code = (packed_val >> (i * 2)) & 0b11;
            
            // Convert code {-1, 0, +1} -> {0, 1, 2}
            float val;
            switch (code) {
                case 0: val = -1.0f; break;
                case 1: val = 0.0f; break;
                case 2: val = 1.0f; break;
                default: val = 0.0f; break;  // Should not happen
            }
            
            w[idx] = val * scale;
        }
    }
    
    return weights;
}

/*
 * Optimized batch pack for multiple tensors
 * Useful for packing all model weights at once
 */
py::list ternary_pack_batch(
    py::list weights_list,
    py::list scales_list,
    bool stochastic = false,
    int seed = 0
) {
    py::list result;
    
    for (size_t i = 0; i < weights_list.size(); i++) {
        py::array_t<float> w = weights_list[i].cast<py::array_t<float>>();
        float scale = scales_list[i].cast<float>();
        
        // Use seed + i for different random sequence per tensor
        auto packed = ternary_pack(w, scale, stochastic, seed + i);
        result.append(packed);
    }
    
    return result;
}

/*
 * Get compression stats
 * Returns theoretical compression ratio
 */
py::dict get_ternary_stats() {
    py::dict stats;
    stats["bits_per_param"] = 2.0;  // 2 bits for 3 values
    stats["compression_ratio_fp32"] = 16.0;  // 32 bits / 2 bits
    stats["compression_ratio_bf16"] = 8.0;   // 16 bits / 2 bits
    stats["values_per_int32"] = 16;
    stats["possible_values"] = py::make_tuple(-1, 0, 1);
    return stats;
}

PYBIND11_MODULE(ternary_codec, m) {
    m.doc() = "Fast ternary weight quantization codec for CHIMERA-M";
    
    m.def("pack", &ternary_pack, 
          "Pack float32 weights to ternary bit format",
          py::arg("weights"), py::arg("scale"), 
          py::arg("stochastic") = false, py::arg("seed") = 0);
    
    m.def("unpack", &ternary_unpack,
          "Unpack ternary bit format back to float32",
          py::arg("packed"), py::arg("original_shape"), py::arg("scale"));
    
    m.def("pack_batch", &ternary_pack_batch,
          "Pack multiple tensors at once",
          py::arg("weights_list"), py::arg("scales_list"),
          py::arg("stochastic") = false, py::arg("seed") = 0);
    
    m.def("get_stats", &get_ternary_stats,
          "Get compression statistics");
}
