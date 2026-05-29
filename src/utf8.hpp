#pragma once

#include <string>
#include <vector>

// Minimal UTF-8 <-> codepoint helpers. misaki operates on Python str (Unicode
// codepoints): vowel/stress/diphthong membership and stress repositioning are
// all per-codepoint, so the phonemizer port must work in codepoints too.
namespace kokoro::utf8 {

inline std::vector<char32_t> decode(const std::string& s) {
  std::vector<char32_t> out;
  out.reserve(s.size());
  std::size_t i = 0;
  while (i < s.size()) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    char32_t cp;
    int extra;
    if (c < 0x80) { cp = c; extra = 0; }
    else if ((c & 0xE0) == 0xC0) { cp = c & 0x1F; extra = 1; }
    else if ((c & 0xF0) == 0xE0) { cp = c & 0x0F; extra = 2; }
    else if ((c & 0xF8) == 0xF0) { cp = c & 0x07; extra = 3; }
    else { ++i; continue; }  // invalid lead byte: skip
    if (i + extra >= s.size()) break;
    bool ok = true;
    for (int k = 0; k < extra; ++k) {
      unsigned char cc = static_cast<unsigned char>(s[i + 1 + k]);
      if ((cc & 0xC0) != 0x80) { ok = false; break; }
      cp = (cp << 6) | (cc & 0x3F);
    }
    if (!ok) { ++i; continue; }
    out.push_back(cp);
    i += extra + 1;
  }
  return out;
}

inline void append(std::string& s, char32_t cp) {
  if (cp < 0x80) {
    s.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    s.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    s.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    s.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    s.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

inline std::string encode(const std::vector<char32_t>& cps) {
  std::string s;
  s.reserve(cps.size());
  for (char32_t cp : cps) append(s, cp);
  return s;
}

} // namespace kokoro::utf8
