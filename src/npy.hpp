// Minimal .npy v1/v2 reader for float32 and float16 arrays.
// Enough for Kokoro voice packs (.npy of shape (510,1,256) float32) and
// per-channel norm stats. C-order only.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace kokoro {

struct NpyArray {
  std::vector<size_t> shape;
  std::vector<float> data; // always materialized as float32

  size_t numel() const {
    size_t n = 1;
    for (auto s : shape) n *= s;
    return n;
  }
};

inline float fp16_to_fp32(uint16_t h) {
  uint32_t sign = (h >> 15) & 1u;
  uint32_t exp  = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu;
  uint32_t out;
  if (exp == 0) {
    if (mant == 0) {
      out = sign << 31;
    } else {
      while (!(mant & 0x400u)) { mant <<= 1; --exp; }
      ++exp; mant &= 0x3FFu;
      out = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
    }
  } else if (exp == 31) {
    out = (sign << 31) | (0xFFu << 23) | (mant << 13);
  } else {
    out = (sign << 31) | ((exp + 112u) << 23) | (mant << 13);
  }
  float r;
  std::memcpy(&r, &out, 4);
  return r;
}

inline NpyArray load_npy(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open npy: " + path);

  char magic[6];
  f.read(magic, 6);
  if (std::memcmp(magic, "\x93NUMPY", 6) != 0)
    throw std::runtime_error("not a .npy file: " + path);

  uint8_t major = 0, minor = 0;
  f.read(reinterpret_cast<char*>(&major), 1);
  f.read(reinterpret_cast<char*>(&minor), 1);

  uint32_t header_len = 0;
  if (major == 1) {
    uint16_t h = 0;
    f.read(reinterpret_cast<char*>(&h), 2);
    header_len = h;
  } else {
    f.read(reinterpret_cast<char*>(&header_len), 4);
  }
  std::string header(header_len, '\0');
  f.read(header.data(), header_len);

  auto find_after = [&](const std::string& key) -> size_t {
    auto p = header.find("'" + key + "'");
    if (p == std::string::npos)
      throw std::runtime_error("npy header missing key: " + key);
    auto colon = header.find(':', p);
    return colon + 1;
  };

  size_t dp = find_after("descr");
  size_t q1 = header.find('\'', dp);
  size_t q2 = header.find('\'', q1 + 1);
  std::string descr = header.substr(q1 + 1, q2 - q1 - 1);
  bool is_fp16 = false;
  size_t elem_bytes = 0;
  if (descr == "<f4" || descr == "|f4") {
    elem_bytes = 4;
  } else if (descr == "<f2" || descr == "|f2") {
    is_fp16 = true;
    elem_bytes = 2;
  } else {
    throw std::runtime_error("unsupported dtype in npy: " + descr);
  }

  size_t sp = find_after("shape");
  size_t t1 = header.find('(', sp);
  size_t t2 = header.find(')', t1);
  std::string shape_str = header.substr(t1 + 1, t2 - t1 - 1);

  NpyArray out;
  size_t i = 0;
  while (i < shape_str.size()) {
    while (i < shape_str.size() && (shape_str[i] == ',' || shape_str[i] == ' '))
      ++i;
    if (i >= shape_str.size()) break;
    char* end = nullptr;
    unsigned long v = std::strtoul(&shape_str[i], &end, 10);
    if (end == &shape_str[i]) break;
    out.shape.push_back(static_cast<size_t>(v));
    i = static_cast<size_t>(end - shape_str.c_str());
  }

  size_t n = out.numel();
  out.data.resize(n);
  if (is_fp16) {
    std::vector<uint16_t> buf(n);
    f.read(reinterpret_cast<char*>(buf.data()),
           static_cast<std::streamsize>(n * elem_bytes));
    for (size_t k = 0; k < n; ++k) out.data[k] = fp16_to_fp32(buf[k]);
  } else {
    f.read(reinterpret_cast<char*>(out.data.data()),
           static_cast<std::streamsize>(n * elem_bytes));
  }
  return out;
}

} // namespace kokoro
