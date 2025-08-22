#pragma once

#include <string.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "common/slice.h"

namespace xllm_service {
constexpr uint32_t MURMUR_HASH3_VALUE_LEN = 16;

struct Murmur3Key {
  uint8_t data[MURMUR_HASH3_VALUE_LEN];

  Murmur3Key() {}
  Murmur3Key(const uint8_t* const input_data) {
    memcpy(data, input_data, MURMUR_HASH3_VALUE_LEN);
  }
  Murmur3Key(const char* const input_data) {
    memcpy(data, input_data, MURMUR_HASH3_VALUE_LEN);
  }

  std::string to_string() const {
    return std::string(reinterpret_cast<const char*>(data),
                       MURMUR_HASH3_VALUE_LEN);
  }

  bool operator==(const Murmur3Key& other) {
    return strncmp(reinterpret_cast<const char*>(data),
                   reinterpret_cast<const char*>(other.data),
                   MURMUR_HASH3_VALUE_LEN);
  }
};

struct FixedStringKeyHash {
  size_t operator()(const Murmur3Key& key) const {
    return std::hash<std::string_view>()(std::string_view(
        reinterpret_cast<const char*>(key.data), sizeof(key.data)));
  }
};

struct FixedStringKeyEqual {
  bool operator()(const Murmur3Key& left, const Murmur3Key& right) const {
    return strncmp(reinterpret_cast<const char*>(left.data),
                   reinterpret_cast<const char*>(right.data),
                   sizeof(left.data)) == 0;
  }
};

void print_hex_array(uint8_t* array);

void murmur_hash3(const uint8_t* pre_hash_value,
                  const Slice<int32_t>& token_ids,
                  uint8_t* hash_value);

}  // namespace xllm_service
