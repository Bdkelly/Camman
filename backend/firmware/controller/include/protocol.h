#pragma once
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace camman {
enum class Kind { Invalid, Hello, Stop, Velocity };
struct Command { Kind kind; float velocity; };

inline Command parse(const char* line) {
    if (std::strcmp(line, "HELLO") == 0) return {Kind::Hello, 0};
    if (std::strcmp(line, "Stop") == 0) return {Kind::Stop, 0};
    if (std::strncmp(line, "V:", 2) != 0) return {Kind::Invalid, 0};
    char* end = nullptr;
    float value = std::strtof(line + 2, &end);
    if (end == line + 2 || *end != '\0' || !std::isfinite(value) || std::fabs(value) > 1)
        return {Kind::Invalid, 0};
    return {Kind::Velocity, value};
}

inline bool expired(unsigned long now, unsigned long last, unsigned long timeout) {
    // Unsigned subtraction remains correct across millis() rollover.
    return static_cast<unsigned long>(now - last) >= timeout;
}
}  // namespace camman
