#!/bin/bash

set -vex

if [[ "$OS_FAMILY" == "windows" ]]; then
	export PATH="/C/Program Files/LLVM/bin:$PATH"
	export LIBCLANG_PATH="/C/Program Files/LLVM/bin"
    export VCPKGRS_DYNAMIC=1
    export VCPKG_ROOT="$HOME/build/vcpkg"
    echo "=== Installed vcpkg packages:"
    "$VCPKG_ROOT/vcpkg" list
fi

echo "=== Current directory: $(pwd)"
echo "=== Environment variable dump:"
export
echo "=== Target settings:"
rustc --print=cfg

if [[ "$FEATURES" == "SCALAR_ONLY" ]]; then
        cargo test -vv --no-default-features
else
    if [[ "$OS_FAMILY" == "osx" ]]; then
        cargo test -vv
    else 
        cargo test -vv --all-features
    fi
fi