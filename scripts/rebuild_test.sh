#!/bin/bash
# rebuild_test.sh - Fast code sync and rebuild for Dynamo runtime container
#
# This script synchronizes code changes from the host-mounted source to the
# container, supporting both Python-only sync (fast) and full Rust rebuilds.
#
# Usage: ./rebuild_test.sh [OPTIONS]
#
# Modes:
#   --fast         Sync only modified Python files (based on git status) [default]
#   --full         Full rebuild: install deps, compile Rust, install Python package
#   --rust         Rebuild Rust code with cargo + maturin (official workflow)
#   --rust-check   Type-check Rust code only (no linking, works on all platforms)
#   --clean        Reinstall package from source
#
# Options:
#   --test         Run tests after sync/build
#   --test-rust    Run Rust tests only
#   --test-python  Run Python tests only
#   --verify       Verify sync worked (import test)
#   --release      Use release profile for Rust builds
#   --help         Show this help
#
# Environment:
#   SOURCE_DIR     Host source directory (default: /home/austin/flow/together-dynamo)
#   DEST_DIR       Container package dir (default: /opt/dynamo/venv/lib/python3.12/site-packages)
#
# Prerequisites for --rust mode:
#   apt-get install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake
#
# ARM Build Note:
#   The full Rust build (--rust) may fail on ARM processors without fullfp16 support.
#   Workaround: Comment out candle-core in lib/llm/Cargo.toml (it's unused).
#   Use --rust-check for type verification without linking.

set -e

# =============================================================================
# CONFIGURATION
# =============================================================================
SOURCE_DIR="${SOURCE_DIR:-/resource/bernyli/together-dynamo}"
DEST_DIR="${DEST_DIR:-/opt/dynamo/venv/lib/python3.12/site-packages}"
COMPONENTS_SRC="${SOURCE_DIR}/components/src/dynamo"
COMPONENTS_DEST="${DEST_DIR}/dynamo"
RUST_VERSION="1.90.0"

# =============================================================================
# ARGUMENT PARSING
# =============================================================================
MODE="fast"   # fast, full, rust, rust-check, clean
RUN_TESTS=false
TEST_RUST=false
TEST_PYTHON=false
VERIFY=false
RELEASE_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            MODE="fast"
            shift
            ;;
        --full)
            MODE="full"
            shift
            ;;
        --rust)
            MODE="rust"
            shift
            ;;
        --rust-check)
            MODE="rust-check"
            shift
            ;;
        --clean)
            MODE="clean"
            shift
            ;;
        --test)
            RUN_TESTS=true
            TEST_RUST=true
            TEST_PYTHON=true
            shift
            ;;
        --test-rust)
            RUN_TESTS=true
            TEST_RUST=true
            shift
            ;;
        --test-python)
            RUN_TESTS=true
            TEST_PYTHON=true
            shift
            ;;
        --verify)
            VERIFY=true
            shift
            ;;
        --release)
            RELEASE_MODE=true
            shift
            ;;
        --help|-h)
            echo "Usage: rebuild_test.sh [OPTIONS]"
            echo ""
            echo "Sync code changes and rebuild for rapid iteration."
            echo ""
            echo "Modes:"
            echo "  --fast         Sync only git-modified Python files (default)"
            echo "  --full         Full rebuild: deps + Rust + Python (fresh container)"
            echo "  --rust         Rebuild Rust code with cargo + maturin"
            echo "  --rust-check   Type-check Rust code only (no linking)"
            echo "  --clean        Reinstall package from source"
            echo ""
            echo "Options:"
            echo "  --test         Run all tests after build"
            echo "  --test-rust    Run Rust tests only"
            echo "  --test-python  Run Python tests only"
            echo "  --verify       Verify sync with import test"
            echo "  --release      Use release profile for Rust builds"
            echo "  --help         Show this help"
            echo ""
            echo "Environment:"
            echo "  SOURCE_DIR     Host source (default: $SOURCE_DIR)"
            echo "  DEST_DIR       Package dir (default: $DEST_DIR)"
            echo ""
            echo "Prerequisites for --rust mode:"
            echo "  apt-get install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake"
            echo ""
            echo "ARM Build Note:"
            echo "  Full Rust build may fail on ARM without fullfp16 support."
            echo "  Workaround: Comment out candle-core in lib/llm/Cargo.toml (unused)."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# SYSTEM DEPENDENCY MANAGEMENT
# =============================================================================

install_system_deps() {
    # Install system dependencies required for Rust build
    # These are needed for: hwloc, udev, protobuf, clang bindings, etc.
    echo ">>> Checking system dependencies..."

    local need_install=false

    # Check for key packages by looking for their libraries/binaries
    if ! pkg-config --exists hwloc 2>/dev/null; then
        need_install=true
    fi
    if ! command -v protoc >/dev/null 2>&1; then
        need_install=true
    fi
    if ! command -v cmake >/dev/null 2>&1; then
        need_install=true
    fi

    if [ "$need_install" = true ]; then
        echo ">>> Installing system build dependencies..."
        apt-get update -qq
        apt-get install -y -qq \
            build-essential \
            libhwloc-dev \
            libudev-dev \
            pkg-config \
            libclang-dev \
            protobuf-compiler \
            python3-dev \
            cmake \
            curl
        echo ">>> System dependencies installed"
    else
        echo ">>> System dependencies already installed"
    fi
}

# =============================================================================
# RUST TOOLCHAIN MANAGEMENT
# =============================================================================

check_rust_installed() {
    command -v cargo >/dev/null 2>&1
}

check_maturin_installed() {
    command -v maturin >/dev/null 2>&1
}

install_rust_toolchain() {
    echo ">>> Installing Rust toolchain..."

    # Install rustup if not present
    if ! command -v rustup >/dev/null 2>&1; then
        echo "  Installing rustup..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none

        # Source cargo environment
        if [ -f "$HOME/.cargo/env" ]; then
            source "$HOME/.cargo/env"
        fi
    fi

    # Install the required Rust version
    echo "  Installing Rust ${RUST_VERSION}..."
    rustup install "${RUST_VERSION}"
    rustup default "${RUST_VERSION}"

    # Verify installation
    if ! check_rust_installed; then
        echo "ERROR: Rust installation failed"
        exit 1
    fi

    echo "  Rust $(rustc --version) installed"
}

install_maturin() {
    echo ">>> Installing maturin..."

    # Check if pip/uv is available
    if command -v uv >/dev/null 2>&1; then
        uv pip install maturin
    elif command -v pip >/dev/null 2>&1; then
        pip install maturin
    else
        echo "ERROR: Neither uv nor pip available to install maturin"
        exit 1
    fi

    if ! check_maturin_installed; then
        echo "ERROR: maturin installation failed"
        exit 1
    fi

    echo "  maturin $(maturin --version) installed"
}

ensure_rust_toolchain() {
    local need_rust=false
    local need_maturin=false

    if ! check_rust_installed; then
        need_rust=true
    fi

    if ! check_maturin_installed; then
        need_maturin=true
    fi

    if [ "$need_rust" = true ] || [ "$need_maturin" = true ]; then
        echo "=== Rust Toolchain Setup ==="

        if [ "$need_rust" = true ]; then
            install_rust_toolchain
        else
            echo ">>> Rust already installed: $(rustc --version)"
        fi

        if [ "$need_maturin" = true ]; then
            install_maturin
        else
            echo ">>> maturin already installed: $(maturin --version)"
        fi

        echo ""
    fi
}

setup_rust_path() {
    # Add Rust to PATH if installed in non-standard location
    # Container has Rust in /home/dynamo/.cargo/bin
    if [ -d "/home/dynamo/.cargo/bin" ]; then
        export PATH="/home/dynamo/.cargo/bin:$PATH"
    fi
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
}

# =============================================================================
# VALIDATION
# =============================================================================
echo "=== Dynamo Rebuild ==="
echo "Mode: ${MODE}"

if [ "$MODE" = "rust" ] || [ "$MODE" = "rust-check" ]; then
    echo "Profile: $([ "$RELEASE_MODE" = true ] && echo "Release" || echo "Dev")"
    echo "Source: ${SOURCE_DIR}"
else
    echo "Source: ${COMPONENTS_SRC}"
    echo "Dest: ${COMPONENTS_DEST}"
fi
echo ""

if [ ! -d "${SOURCE_DIR}" ]; then
    echo "ERROR: Source directory not found: ${SOURCE_DIR}"
    echo "Is the host directory mounted?"
    exit 1
fi

if [ "$MODE" != "rust" ] && [ "$MODE" != "rust-check" ] && [ ! -d "${DEST_DIR}" ]; then
    echo "ERROR: Destination directory not found: ${DEST_DIR}"
    echo "Is the Python environment activated?"
    exit 1
fi

# =============================================================================
# SYNC FUNCTIONS
# =============================================================================

sync_modified_files() {
    # Sync only files modified according to git status
    echo ">>> Syncing modified Python files..."
    local count=0
    local skipped=0

    cd "${SOURCE_DIR}"

    # Get modified Python files from components/src/dynamo
    # Use process substitution to avoid subshell and preserve count
    while read f; do
        if [ -z "$f" ]; then
            continue
        fi
        if [ -f "$f" ] && [[ "$f" == *.py ]]; then
            # Extract relative path within dynamo package
            rel_path="${f#components/src/dynamo/}"
            dest="${COMPONENTS_DEST}/${rel_path}"
            dest_dir="$(dirname "$dest")"

            # Check if file changed (md5 comparison)
            if [ ! -f "$dest" ] || [ "$(md5sum "$f" | awk '{print $1}')" != "$(md5sum "$dest" 2>/dev/null | awk '{print $1}')" ]; then
                mkdir -p "$dest_dir" 2>/dev/null || true
                if cp "$f" "$dest" 2>/dev/null; then
                    echo "  + ${rel_path}"
                    count=$((count + 1))
                else
                    echo "  ! ${rel_path} (copy failed)"
                fi
            else
                echo "  = ${rel_path} (unchanged)"
                skipped=$((skipped + 1))
            fi
        fi
    done < <(git status --porcelain -uno components/src/dynamo | awk '{print $NF}')

    echo ">>> Synced ${count} file(s), skipped ${skipped} unchanged"
}

sync_all_files() {
    # Full rsync of all Python files
    echo ">>> Full sync of Python files..."

    rsync -av --include='*/' --include='*.py' --exclude='*' \
        "${COMPONENTS_SRC}/" "${COMPONENTS_DEST}/" \
        --checksum --delete

    echo ">>> Full sync complete"
}

full_rebuild() {
    # Full rebuild: system deps, Rust toolchain, Rust bindings, Python package
    # This is the "works anywhere" mode for fresh containers
    echo "=== Full Rebuild Mode ==="
    local start_time=$(date +%s)

    # Step 0: Install system dependencies if missing
    install_system_deps

    # Step 1: Setup Rust toolchain
    setup_rust_path
    ensure_rust_toolchain

    # Step 2: Build Rust bindings with maturin develop
    echo ">>> Building Python bindings with maturin..."
    cd "${SOURCE_DIR}/lib/bindings/python"

    # Always use release for full rebuild (production-like)
    if command -v uv >/dev/null 2>&1; then
        maturin develop --uv --release
    else
        maturin develop --release
    fi

    # Step 3: Install Python package in editable mode
    install_python_package

    local elapsed=$(($(date +%s) - start_time))
    echo ">>> Full rebuild completed in ${elapsed}s"
}

clean_reinstall() {
    # Full reinstall from source
    echo ">>> Clean reinstall from source..."

    cd "${SOURCE_DIR}"
    pip install -e . --no-deps --force-reinstall

    echo ">>> Reinstall complete"
}

# =============================================================================
# RUST BUILD FUNCTIONS (Official Workflow)
# =============================================================================

build_rust_bindings() {
    # Build Rust Python bindings using official workflow:
    # 1. cd lib/bindings/python
    # 2. maturin develop --uv
    echo ">>> Building Python bindings with maturin..."

    cd "${SOURCE_DIR}/lib/bindings/python"

    local release_arg=""
    if [ "$RELEASE_MODE" = true ]; then
        release_arg="--release"
    fi

    # Use maturin develop to install bindings into current venv
    if command -v uv >/dev/null 2>&1; then
        maturin develop --uv ${release_arg}
    else
        maturin develop ${release_arg}
    fi

    echo ">>> Python bindings build complete"
}

install_python_package() {
    # Install the full Python package in editable mode
    echo ">>> Installing Python package..."

    cd "${SOURCE_DIR}"

    if command -v uv >/dev/null 2>&1; then
        uv pip install -e .
    else
        pip install -e .
    fi

    echo ">>> Python package installed"
}

check_rust() {
    # Type-check Rust code without linking
    # This is useful for verifying code changes on platforms where full
    # linking fails due to architecture-specific issues (e.g., ARM fp16)
    echo ">>> Type-checking Rust workspace..."

    cd "${SOURCE_DIR}"

    # Determine profile
    local profile_arg="--profile dev"
    if [ "$RELEASE_MODE" = true ]; then
        profile_arg="--release"
    fi

    # Check the dynamo-llm crate (contains kv_router and routing strategies)
    cargo check -p dynamo-llm ${profile_arg}

    echo ">>> Rust type-check complete"
}

rust_rebuild() {
    # Full Rust rebuild workflow (official process)
    # First ensure system deps are installed (protoc, etc.)
    install_system_deps

    setup_rust_path
    ensure_rust_toolchain

    echo ">>> Starting full Rust rebuild..."
    local start_time=$(date +%s)

    # Step 1: Build Rust bindings with maturin develop
    build_rust_bindings

    # Step 2: Install Python package in editable mode
    install_python_package

    local elapsed=$(($(date +%s) - start_time))
    echo ">>> Full Rust rebuild completed in ${elapsed}s"
}

rust_check_only() {
    # Rust type-check workflow (no linking, no Python bindings)
    # Useful on platforms where full linking fails (e.g., ARM without fp16)
    # First ensure system deps are installed (protoc, etc.)
    install_system_deps

    setup_rust_path
    ensure_rust_toolchain

    check_rust
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================
START_TIME=$(date +%s)

case $MODE in
    fast)
        sync_modified_files
        ;;
    full)
        full_rebuild
        ;;
    rust)
        rust_rebuild
        ;;
    rust-check)
        rust_check_only
        ;;
    clean)
        clean_reinstall
        ;;
esac

BUILD_TIME=$(($(date +%s) - START_TIME))
echo ""
echo ">>> Build completed in ${BUILD_TIME}s"

# =============================================================================
# VERIFICATION
# =============================================================================
if [ "$VERIFY" = true ]; then
    echo ""
    echo "=== Verification ==="
    echo ">>> Testing imports..."

    python3 -c "
import dynamo
import dynamo.frontend
import dynamo.trtllm
print('All imports successful')
print(f'dynamo version: {dynamo.__version__ if hasattr(dynamo, \"__version__\") else \"unknown\"}')
"

    echo ">>> Verification passed"
fi

# =============================================================================
# TESTS
# =============================================================================
if [ "$RUN_TESTS" = true ]; then
    echo ""
    echo "=== Running Tests ==="

    if [ "$TEST_RUST" = true ]; then
        echo ">>> Running Rust tests..."
        cd "${SOURCE_DIR}"

        setup_rust_path

        # Run tests for dynamo-llm crate which contains kv_router
        cargo test -p dynamo-llm

        echo ">>> Rust tests complete"
    fi

    if [ "$TEST_PYTHON" = true ]; then
        echo ">>> Running Python tests..."
        cd "${SOURCE_DIR}"
        pytest tests/ -x -q --tb=short
        echo ">>> Python tests complete"
    fi
fi

echo ""
echo "=== Done ==="
