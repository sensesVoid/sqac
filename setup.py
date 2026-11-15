import os

from setuptools import setup

extensions = []
if os.environ.get("SQAC_SKIP_RUST") != "1":
    from setuptools_rust import RustExtension

    extensions.append(
        RustExtension(
            "sqac._simd",
            path="sqac-simd/Cargo.toml",
            optional=True,
            py_limited_api=True,
        )
    )

setup(
    rust_extensions=extensions,
    options={"bdist_wheel": {"py_limited_api": "cp310"}},
)
