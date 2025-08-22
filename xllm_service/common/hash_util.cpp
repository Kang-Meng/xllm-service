
#include "common/hash_util.h"

#include <MurmurHash3.h>
#include <assert.h>

#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>

#include "common/global_gflags.h"

namespace xllm_service {

void murmur_hash3(const uint8_t* pre_hash_value,
                  const Slice<int32_t>& token_ids,
                  uint8_t* hash_value) {
  if (pre_hash_value == nullptr) {
    MurmurHash3_x64_128(reinterpret_cast<const void*>(token_ids.data()),
                        sizeof(int32_t) * token_ids.size(),
                        FLAGS_murmur_hash3_seed,
                        hash_value);
  } else {
    uint8_t key[1024];

    int32_t data_len =
        sizeof(int32_t) * token_ids.size() + MURMUR_HASH3_VALUE_LEN;
    assert(sizeof(key) > data_len);

    memcpy(key, pre_hash_value, MURMUR_HASH3_VALUE_LEN);
    memcpy(key + MURMUR_HASH3_VALUE_LEN,
           reinterpret_cast<const void*>(token_ids.data()),
           sizeof(int32_t) * token_ids.size());

    // print_hex_array(key, data_len);
    MurmurHash3_x64_128(reinterpret_cast<const void*>(key),
                        data_len,
                        FLAGS_murmur_hash3_seed,
                        hash_value);
  }
}

void print_hex_array(uint8_t* array) {
  for (size_t i = 0; i < MURMUR_HASH3_VALUE_LEN; ++i) {
    unsigned char uc = static_cast<unsigned char>(array[i]);
    std::cout << std::hex << std::setw(2) << std::setfill('0')
              << static_cast<int>(uc);

    if (i % MURMUR_HASH3_VALUE_LEN == MURMUR_HASH3_VALUE_LEN - 1) {
      std::cout << std::endl;
    }

    else {
      std::cout << " ";
    }
  }
  std::cout << std::dec << std::endl;
}

}  // namespace xllm_service