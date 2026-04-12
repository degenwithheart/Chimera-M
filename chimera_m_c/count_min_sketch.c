/*
 * Count-Min Sketch C Extension for CHIMERA-M
 * Fast hash-based optimizer state compression
 * 
 * Compile:
 *   cd chimera_m_c
 *   gcc -O3 -shared -fPIC -o count_min_sketch.so count_min_sketch.c -lm
 * 
 * macOS:
 *   gcc -O3 -shared -fPIC -arch x86_64 -arch arm64 -o count_min_sketch.so count_min_sketch.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/*
 * Hash function for Count-Min Sketch
 * Universal hash: (a * x + b) % p % width
 * 
 * Using a simple but fast hash suitable for CMS
 */
static inline uint32_t hash_index(uint32_t index, uint32_t seed, int width) {
    // Universal hash parameters
    uint32_t a = seed * 2 + 1;  // Must be odd
    uint32_t b = seed / 2;
    uint32_t p = 2147483647u;   // Large prime (2^31 - 1)
    
    uint64_t prod = (uint64_t)a * index + b;
    uint32_t hash = (uint32_t)(prod % p);
    return hash % width;
}

/*
 * Update Count-Min Sketch tables with momentum and variance values
 * 
 * Args:
 *   tables_m: float array [depth][width] - momentum table (updated in-place)
 *   tables_v: float array [depth][width] - variance table (updated in-place)
 *   indices: int32 array [n] - parameter indices to update
 *   values_m: float array [n] - momentum values
 *   values_v: float array [n] - variance values
 *   seeds: int32 array [depth] - hash seeds per row
 *   depth: int - number of hash rows
 *   width: int - number of buckets per row
 *   n: int - number of indices
 *   beta1: float - Adam momentum decay
 *   beta2: float - Adam variance decay
 *   step_count: int - current step for bias correction
 */
void cms_update(
    float* tables_m,
    float* tables_v,
    const int32_t* indices,
    const float* values_m,
    const float* values_v,
    const int32_t* seeds,
    int depth,
    int width,
    int n,
    float beta1,
    float beta2,
    int step_count
) {
    // Bias correction (Adam-style)
    float bias_correction1 = 1.0f - powf(beta1, (float)step_count);
    float bias_correction2 = 1.0f - powf(beta2, (float)step_count);
    
    // Update each index
    for (int i = 0; i < n; i++) {
        uint32_t idx = (uint32_t)indices[i];
        float m_val = values_m[i] / bias_correction1;
        float v_val = values_v[i] / bias_correction2;
        
        // Update each hash row
        for (int d = 0; d < depth; d++) {
            uint32_t bucket = hash_index(idx, seeds[d], width);
            size_t offset = d * width + bucket;
            
            // Adam-style EMA update
            tables_m[offset] = beta1 * tables_m[offset] + (1.0f - beta1) * m_val;
            tables_v[offset] = beta2 * tables_v[offset] + (1.0f - beta2) * v_val;
        }
    }
}

/*
 * Query Count-Min Sketch for momentum and variance estimates
 * Returns minimum across all hash rows (conservative estimate)
 * 
 * Args:
 *   tables_m: float array [depth][width] - momentum table
 *   tables_v: float array [depth][width] - variance table
 *   indices: int32 array [n] - parameter indices to query
 *   seeds: int32 array [depth] - hash seeds per row
 *   depth: int
 *   width: int
 *   n: int
 *   out_m: float array [n] - output momentum estimates (min across rows)
 *   out_v: float array [n] - output variance estimates (min across rows)
 */
void cms_query(
    const float* tables_m,
    const float* tables_v,
    const int32_t* indices,
    const int32_t* seeds,
    int depth,
    int width,
    int n,
    float* out_m,
    float* out_v
) {
    // Query each index
    for (int i = 0; i < n; i++) {
        uint32_t idx = (uint32_t)indices[i];
        
        float min_m = INFINITY;
        float min_v = INFINITY;
        
        // Check all hash rows
        for (int d = 0; d < depth; d++) {
            uint32_t bucket = hash_index(idx, seeds[d], width);
            size_t offset = d * width + bucket;
            
            float m_val = tables_m[offset];
            float v_val = tables_v[offset];
            
            if (m_val < min_m) min_m = m_val;
            if (v_val < min_v) min_v = v_val;
        }
        
        out_m[i] = min_m;
        out_v[i] = min_v;
    }
}

/*
 * Batch update variant for larger batches with SIMD-friendly access pattern
 * Processes depth rows sequentially for better cache utilization
 */
void cms_update_batch_optimized(
    float* tables_m,
    float* tables_v,
    const int32_t* indices,
    const float* values_m,
    const float* values_v,
    const int32_t* seeds,
    int depth,
    int width,
    int n,
    float beta1,
    float beta2,
    int step_count
) {
    // Bias correction
    float bias_correction1 = 1.0f - powf(beta1, (float)step_count);
    float bias_correction2 = 1.0f - powf(beta2, (float)step_count);
    
    // Pre-compute bias-corrected values
    float* corrected_m = (float*)malloc(n * sizeof(float));
    float* corrected_v = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        corrected_m[i] = values_m[i] / bias_correction1;
        corrected_v[i] = values_v[i] / bias_correction2;
    }
    
    // Update row by row for better cache locality
    for (int d = 0; d < depth; d++) {
        float* row_m = tables_m + d * width;
        float* row_v = tables_v + d * width;
        uint32_t seed = seeds[d];
        
        for (int i = 0; i < n; i++) {
            uint32_t bucket = hash_index(indices[i], seed, width);
            
            row_m[bucket] = beta1 * row_m[bucket] + (1.0f - beta1) * corrected_m[i];
            row_v[bucket] = beta2 * row_v[bucket] + (1.0f - beta2) * corrected_v[i];
        }
    }
    
    free(corrected_m);
    free(corrected_v);
}

/*
 * Initialize tables to zero
 */
void cms_init_tables(float* tables_m, float* tables_v, int depth, int width) {
    size_t total = depth * width;
    memset(tables_m, 0, total * sizeof(float));
    memset(tables_v, 0, total * sizeof(float));
}

/*
 * Get memory usage statistics
 */
void cms_get_stats(int depth, int width, size_t* table_bytes, size_t* total_bytes) {
    size_t one_table = depth * width * sizeof(float);
    *table_bytes = one_table;
    *total_bytes = one_table * 2;  // m and v tables
}
