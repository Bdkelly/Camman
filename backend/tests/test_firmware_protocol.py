"""Compile/run the actual portable parser used by the ESP32 firmware."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_native_firmware_parser_and_watchdog(tmp_path):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("Install g++ to run the native firmware protocol test")
    include = Path(__file__).resolve().parents[1] / "firmware" / "controller" / "include"
    source = tmp_path / "protocol_test.cpp"
    source.write_text(
        r"""
#include <cassert>
#include <limits>
#include "protocol.h"
int main() {
    using namespace camman;
    assert(parse("HELLO").kind == Kind::Hello);
    assert(parse("Stop").kind == Kind::Stop);
    assert(parse("V:0.2500").velocity == 0.25f);
    assert(parse("V:-1.0000").velocity == -1.0f);
    for (const char* invalid : {"", "V:", "V:1.01", "V:-1.01", "V:nan", "V:inf", "V:0.2junk", "P:1,0", "Right"})
        assert(parse(invalid).kind == Kind::Invalid);
    assert(!expired(1000, 500, 750));
    assert(expired(1250, 500, 750));
    const auto last = std::numeric_limits<unsigned long>::max() - 100;
    assert(!expired(100, last, 750));
    assert(expired(800, last, 750));
}
""".replace("#include <limits>", "#include <limits>\n#include <initializer_list>")
    )
    binary = tmp_path / "protocol_test"
    subprocess.run(
        [
            compiler,
            "-std=c++11",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(include),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
