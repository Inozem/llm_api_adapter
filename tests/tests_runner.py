import pytest

if __name__ == "__main__":
    unit_result = pytest.main([
        "-v",
        "--rootdir=.",
        "--pyargs",
        "unit"
    ])
    if unit_result == 0:
        integ_result = pytest.main([
            "-v",
            "--rootdir=.",
            "--pyargs",
            "integration"
        ])
        exit(integ_result)
    else:
        exit(unit_result)
