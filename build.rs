use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // Tell cargo to invalidate the built crate whenever wrapper files change
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=LLAMA_CUDA");
    println!("cargo:rerun-if-env-changed=LLAMA_METAL");
    println!("cargo:rerun-if-env-changed=LLAMA_HIPBLAS");
    println!("cargo:rerun-if-env-changed=LLAMA_CLBLAST");
    // CPU optimization knobs consumed in build_llama_cpp(). Without these,
    // cargo does not re-run build.rs when only these env vars change, so the
    // knobs silently fail to take effect (the cached llama.cpp artifact is
    // reused).
    println!("cargo:rerun-if-env-changed=MULLAMA_NO_NATIVE");
    println!("cargo:rerun-if-env-changed=MULLAMA_LTO");
    println!("cargo:rerun-if-env-changed=MULLAMA_OPENMP");
    println!("cargo:rerun-if-env-changed=MULLAMA_STATIC");

    // Set up platform-specific configurations
    setup_platform_specific();

    // Print dependency errors if needed
    print_dependency_errors();

    // Determine the path to llama.cpp
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let llama_cpp_path = manifest_dir.join("llama.cpp");

    // Check if llama.cpp exists, if not, we can't proceed
    if !llama_cpp_path.exists() {
        eprintln!("WARNING: llama.cpp not found at {:?}", llama_cpp_path);
        eprintln!("This crate requires the llama.cpp source code to build.");
        eprintln!("Please either:");
        eprintln!("1. Clone this repository with submodules: git clone --recurse-submodules");
        eprintln!("2. Initialize submodules: git submodule update --init --recursive");
        eprintln!("3. Set LLAMA_CPP_PATH environment variable to point to a llama.cpp checkout");
        return;
    }

    // Check if the llama.cpp directory has the required files
    if !llama_cpp_path.join("include").join("llama.h").exists() {
        eprintln!("WARNING: llama.h not found in llama.cpp include directory");
        return;
    }

    // Build the C++ library using CMake
    let dst = build_llama_cpp(&llama_cpp_path);

    // Generate bindings
    generate_bindings(&llama_cpp_path, &dst);
}

fn setup_platform_specific() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    match target_os.as_str() {
        "windows" => setup_windows(),
        "macos" => setup_macos(&target_arch),
        "linux" => setup_linux(),
        _ => println!("cargo:warning=Unsupported target OS: {}", target_os),
    }
}

fn setup_windows() {
    println!("cargo:rustc-cfg=target_platform=\"windows\"");

    // Link Windows-specific libraries
    println!("cargo:rustc-link-lib=ole32");
    println!("cargo:rustc-link-lib=oleaut32");
    println!("cargo:rustc-link-lib=winmm");
    println!("cargo:rustc-link-lib=dsound");
    println!("cargo:rustc-link-lib=dxguid");
    println!("cargo:rustc-link-lib=user32");
    println!("cargo:rustc-link-lib=kernel32");

    // Check for Visual Studio
    if let Ok(vs_path) = env::var("VCINSTALLDIR") {
        println!("cargo:rustc-link-search=native={}/lib/x64", vs_path);
    }

    // Windows-specific compiler flags
    if env::var("PROFILE").unwrap() == "release" {
        println!("cargo:rustc-env=CFLAGS=/O2 /GL /DNDEBUG");
        println!("cargo:rustc-env=CXXFLAGS=/O2 /GL /DNDEBUG");
    }
}

fn setup_macos(target_arch: &str) {
    println!("cargo:rustc-cfg=target_platform=\"macos\"");

    // Link macOS frameworks
    println!("cargo:rustc-link-lib=framework=CoreAudio");
    println!("cargo:rustc-link-lib=framework=AudioToolbox");
    println!("cargo:rustc-link-lib=framework=AudioUnit");
    println!("cargo:rustc-link-lib=framework=CoreFoundation");
    println!("cargo:rustc-link-lib=framework=CoreServices");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Apple Silicon specific optimizations
    if target_arch == "aarch64" {
        println!("cargo:rustc-cfg=target_arch_apple_silicon");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=MetalKit");

        // Enable Metal by default on Apple Silicon
        if env::var("LLAMA_METAL").is_err() {
            env::set_var("LLAMA_METAL", "1");
        }
    }

    // macOS-specific compiler flags
    if env::var("PROFILE").unwrap() == "release" {
        if target_arch == "aarch64" {
            println!("cargo:rustc-env=CFLAGS=-O3 -mcpu=apple-m1");
            println!("cargo:rustc-env=CXXFLAGS=-O3 -mcpu=apple-m1");
        } else {
            println!("cargo:rustc-env=CFLAGS=-O3 -march=native");
            println!("cargo:rustc-env=CXXFLAGS=-O3 -march=native");
        }
    }
}

fn setup_linux() {
    println!("cargo:rustc-cfg=target_platform=\"linux\"");

    // Check for audio libraries using pkg-config
    check_audio_libraries();

    // Linux-specific compiler flags
    if env::var("PROFILE").unwrap() == "release" {
        println!("cargo:rustc-env=CFLAGS=-O3 -march=native -mtune=native -DNDEBUG");
        println!("cargo:rustc-env=CXXFLAGS=-O3 -march=native -mtune=native -DNDEBUG");
    }

    // Check for NUMA support
    if pkg_config::probe_library("numa").is_ok() {
        println!("cargo:rustc-cfg=feature=\"numa\"");
        println!("cargo:rustc-link-lib=numa");
    }

    // Standard Linux libraries
    println!("cargo:rustc-link-lib=pthread");
    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");
}

fn check_audio_libraries() {
    // Check for ALSA
    if pkg_config::probe_library("alsa").is_ok() {
        println!("cargo:rustc-cfg=feature=\"alsa\"");
        println!("cargo:rustc-link-lib=asound");
    } else {
        println!("cargo:warning=ALSA development libraries not found. Install libasound2-dev");
    }

    // Check for PulseAudio
    if pkg_config::probe_library("libpulse").is_ok() {
        println!("cargo:rustc-cfg=feature=\"pulseaudio\"");
        println!("cargo:rustc-link-lib=pulse");
    } else {
        println!("cargo:warning=PulseAudio development libraries not found. Install libpulse-dev");
    }

    // Check for JACK
    if pkg_config::probe_library("jack").is_ok() {
        println!("cargo:rustc-cfg=feature=\"jack\"");
        println!("cargo:rustc-link-lib=jack");
    }

    // Check for additional audio libraries
    for lib in &["flac", "vorbis", "vorbisenc", "opus"] {
        if pkg_config::probe_library(lib).is_ok() {
            println!("cargo:rustc-cfg=feature=\"{}\"", lib);
        }
    }
}

fn build_llama_cpp(llama_cpp_path: &PathBuf) -> PathBuf {
    let mut cmake_config = cmake::Config::new(llama_cpp_path);

    // Set build type
    if env::var("PROFILE").unwrap() == "release" {
        cmake_config.define("CMAKE_BUILD_TYPE", "Release");
    } else {
        cmake_config.define("CMAKE_BUILD_TYPE", "Debug");
    }

    // Platform-specific CMake configurations
    if cfg!(target_os = "windows") {
        cmake_config.define("CMAKE_GENERATOR_PLATFORM", "x64");
        cmake_config.define("CMAKE_MSVC_RUNTIME_LIBRARY", "MultiThreadedDLL");
    }

    // GPU acceleration configurations
    if env::var("LLAMA_CUDA").is_ok() {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        cmake_config.define("LLAMA_CUDA", "ON");
        cmake_config.define("CMAKE_CUDA_ARCHITECTURES", "61;70;75;80;86;89");
        configure_cuda_linking();
    } else {
        cmake_config.define("LLAMA_CUDA", "OFF");
    }

    if env::var("LLAMA_METAL").is_ok() {
        println!("cargo:rustc-cfg=feature=\"metal\"");
        cmake_config.define("LLAMA_METAL", "ON");
    } else {
        cmake_config.define("LLAMA_METAL", "OFF");
    }

    if env::var("LLAMA_HIPBLAS").is_ok() {
        println!("cargo:rustc-cfg=feature=\"rocm\"");
        cmake_config.define("LLAMA_HIPBLAS", "ON");
        configure_rocm_linking();
    } else {
        cmake_config.define("LLAMA_HIPBLAS", "OFF");
    }

    if env::var("LLAMA_CLBLAST").is_ok() {
        println!("cargo:rustc-cfg=feature=\"opencl\"");
        cmake_config.define("LLAMA_CLBLAST", "ON");
        configure_opencl_linking();
    } else {
        cmake_config.define("LLAMA_CLBLAST", "OFF");
    }

    // --- Build mode: shared backends (default) vs static ---
    //
    // Two ways to host llama.cpp:
    //
    //  * **Static** (`MULLAMA_STATIC=1`): llama.cpp is compiled into static
    //    archives and linked into the Rust binary. Simple, one self-contained
    //    binary, no runtime .so hunting. But the final link goes through
    //    rust-lld, which *cannot* consume GCC LTO bitcode, so `GGML_LTO` is off
    //    — and the hot Q4 vec_dot kernels can't be inlined into the matmul
    //    loop. This leaves a ~1.17x decode gap vs ollama.
    //
    //  * **Shared backends** (default on Linux x86_64, no GPU): build llama.cpp
    //    as shared libraries with `GGML_BACKEND_DL` + `GGML_CPU_ALL_VARIANTS` —
    //    exactly ollama's model. The per-microarch CPU backend
    //    (`libggml-cpu-alderlake.so`, `libggml-cpu-haswell.so`, ...) is built as
    //    a *separate* .so, linked by GCC with LTO (`GGML_LTO=ON` is now safe —
    //    rust-lld only links the native `libllama.so` dylib, no GCC bitcode
    //    crosses the Rust link boundary), and dlopen'd at runtime by
    //    `ggml_backend_load_all()` (triggered by the `llama_backend_init()`
    //    call in src/lib.rs). The loader scores each backend against the host
    //    CPU and picks the best (e.g. `alderlake` on an i9-12950HX), so the
    //    binary is also portable across x86 microarches — a win over the static
    //    `-march=native` build. `GGML_NATIVE` is *incompatible* with
    //    `GGML_BACKEND_DL` (cmake FATAL_ERROR), so it is OFF here; the variants
    //    carry their own ISA flags.
    let gpu_backend = env::var("LLAMA_CUDA").is_ok()
        || env::var("LLAMA_METAL").is_ok()
        || env::var("LLAMA_HIPBLAS").is_ok()
        || env::var("LLAMA_CLBLAST").is_ok();
    let shared = env::var("MULLAMA_STATIC").is_err()
        && cfg!(target_os = "linux")
        && cfg!(target_arch = "x86_64")
        && !gpu_backend;

    // General CPU optimizations.
    //
    // NOTE: upstream llama.cpp renamed the `LLAMA_*` cmake options to `GGML_*`.
    // Only a few (LLAMA_NATIVE/CUDA/METAL/...) retain a deprecation shim that
    // maps them to the new name; `LLAMA_AVX`, `LLAMA_AVX2`, `LLAMA_FMA`,
    // `LLAMA_F16C`, `LLAMA_LTO`, `LLAMA_OPENMP` are *dead* options that this
    // build used to pass, so they were silently ignored. The CMakeCache of a
    // release build confirmed it: `GGML_LTO=OFF` despite `LLAMA_LTO=ON`, and
    // `GGML_AVX/AVX2/FMA/F16C=OFF` (SIMD still reached the hot kernels only
    // because `LLAMA_NATIVE` maps to `GGML_NATIVE`, which drives
    // `-march=native` on the ggml-cpu target). Use the real `GGML_*` names.
    //
    // In STATIC+native mode `GGML_NATIVE` adds `-march=native` to the
    // *ggml-cpu* target's ARCH_FLAGS only; some TUs live in `ggml-base` and
    // would miss it, so we ALSO inject `-march=native -O3` via CMAKE_C/CXX_FLAGS
    // so every TU compiles with the host's full ISA + -O3. Set
    // MULLAMA_NO_NATIVE=1 for a portable baseline. In SHARED mode GGML_NATIVE
    // is incompatible with GGML_BACKEND_DL (OFF) and the per-variant .so's
    // carry their own ISA, so we only inject -O3.
    if shared {
        cmake_config.define("GGML_NATIVE", "OFF");
        for flag in ["-O3"] {
            cmake_config.cflag(flag);
            cmake_config.cxxflag(flag);
        }
    } else if env::var("MULLAMA_NO_NATIVE").is_err() {
        cmake_config.define("GGML_NATIVE", "ON");
        for flag in ["-O3", "-march=native", "-mtune=native"] {
            cmake_config.cflag(flag);
            cmake_config.cxxflag(flag);
        }
    } else {
        cmake_config.define("GGML_NATIVE", "OFF");
    }

    // Explicit SIMD flags. In STATIC non-native mode these select kernels
    // (gated on GGML_AVX/AVX2/FMA/F16C defines); in native mode they are
    // inert (the kernels gate on the -march=native compiler macros instead);
    // in SHARED mode they are inert (ALL_VARIANTS sets them per-variant).
    cmake_config.define("GGML_AVX", "ON");
    cmake_config.define("GGML_AVX2", "ON");
    cmake_config.define("GGML_FMA", "ON");
    cmake_config.define("GGML_F16C", "ON");

    // Link-Time Optimization: lets the compiler inline the quantized vec_dot
    // kernels into the matmul loop — the main lever for closing the decode gap.
    //
    // In SHARED mode it is always ON and *safe*: the CPU backend .so is linked
    // by GCC (LTO bitcode resolved entirely within the gcc link step), and
    // rust-lld only links the finished native `libllama.so` dylib — no GCC
    // bitcode crosses the Rust link boundary (the failure mode that made LTO
    // impossible in STATIC mode).
    //
    // In STATIC mode LTO is OFF by default: rust-lld cannot consume GCC LTO
    // bitcode (`rust-lld: error: too many errors`). Set MULLAMA_LTO=1 to
    // attempt it (will fail to link unless the final link is driven by a GCC
    // linker with `-ffat-lto-objects`).
    if shared || env::var("MULLAMA_LTO").is_ok() {
        cmake_config.define("GGML_LTO", "ON");
    } else {
        cmake_config.define("GGML_LTO", "OFF");
    }

    // CPU-backend threading. OpenMP is OFF by default — we use ggml's internal
    // threadpool, which is what modern llama.cpp and ollama's shipped runners
    // use. OpenMP's per-op fork-join + barrier overhead hurts throughput on
    // small models where each matmul is tiny: measured on qwen2.5-0.5b Q4_K_M
    // (i9-12950HX), OpenMP-on peaked at ~52 tok/s @ 6 threads while OpenMP-off
    // scaled to ~70 tok/s @ 10-12 threads — a ~35% win and the gap to ollama
    // closed from ~1.55x to ~1.15x. It is also at least as fast on larger
    // models, where per-op overhead is relatively smaller. Set MULLAMA_OPENMP=1
    // to restore the legacy OpenMP backend (A/B / fallback).
    let openmp = env::var("MULLAMA_OPENMP").is_ok();
    if openmp {
        cmake_config.define("GGML_OPENMP", "ON");
    } else {
        cmake_config.define("GGML_OPENMP", "OFF");
    }

    // Build configuration
    cmake_config.define("LLAMA_BUILD_TESTS", "OFF");
    cmake_config.define("LLAMA_BUILD_EXAMPLES", "OFF");

    if shared {
        // Shared-backend build (ollama's model).
        cmake_config.define("BUILD_SHARED_LIBS", "ON");
        cmake_config.define("LLAMA_STATIC", "OFF");
        cmake_config.define("GGML_BACKEND_DL", "ON");
        // Build every x86 microarch variant; the runtime loader picks the best
        // for the host (alderlake on i9-12950HX). Makes the binary portable.
        cmake_config.define("GGML_CPU_ALL_VARIANTS", "ON");
        // Do NOT set GGML_BACKEND_DIR: it bakes a build-machine path into the
        // binary as a compile-time search dir. We copy backends next to the
        // binary instead (loader scans /proc/self/exe dir + CWD).
    } else {
        cmake_config.define("BUILD_SHARED_LIBS", "OFF");
        cmake_config.define("LLAMA_STATIC", "ON");
    }

    let dst = cmake_config.build();

    if shared {
        link_shared_backends(&dst, openmp);
    } else {
        link_static(&dst, openmp);
    }

    dst
}

fn link_static(dst: &std::path::Path, openmp: bool) {
    // Link the built static archives.
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=llama");

    // Modern llama.cpp splits ggml into multiple libraries.
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=static=ggml_static");
    } else {
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
    }

    link_runtime_libs(openmp);
}

fn link_shared_backends(dst: &std::path::Path, openmp: bool) {
    // Shared-backend build. The cmake install lays out:
    //   {dst}/lib/  -> libllama.so, libggml.so, libggml-base.so (+ version symlinks)
    //   {dst}/bin/  -> libggml-cpu-<variant>.so  (the dlopen'd CPU backends)
    //
    // The Rust binary links libllama.so as a dylib (rust-lld links a native
    // .so — no GCC LTO bitcode crosses the boundary, so GGML_LTO works).
    // libllama.so transitively needs libggml.so / libggml-base.so.
    //
    // At RUNTIME the ggml backend loader (ggml_backend_load_best, called from
    // llama_backend_init) searches for libggml-cpu-*.so in the *executable's
    // directory* and CWD — NOT rpath, NOT the libggml.so dir. So the CPU
    // backend .so's MUST sit next to the mullama binary. We copy them (and the
    // runtime .so's) into the cargo target/<profile>/ dir, where the binary
    // lives, and set rpath=$ORIGIN so the dynamic linker finds libllama.so /
    // libggml.so there too.
    let lib_dir = dst.join("lib");
    let bin_dir = dst.join("bin");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    // libllama.so's NEEDED entries pull in libggml.so + libggml-base.so; we
    // list them explicitly so the link search is unambiguous.
    println!("cargo:rustc-link-lib=dylib=ggml");
    println!("cargo:rustc-link-lib=dylib=ggml-base");

    // rpath: $ORIGIN so the binary finds the .so's bundled next to it (the
    // copy step below puts them there), plus the absolute build dir as a
    // dev fallback so `cargo run` works even before/without the copy.
    //
    // We use DT_RPATH (--disable-new-dtags), NOT DT_RUNPATH: RPATH is searched
    // transitively, so the executable's rpath resolves libllama.so's own
    // NEEDED entries (libggml.so / libggml-base.so). DT_RUNPATH is *not*
    // transitive, which left libggml.so "not found" at runtime.
    println!("cargo:rustc-link-arg=-Wl,--disable-new-dtags");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());

    // Copy every .so (runtime libs from lib/, CPU backends from bin/) next to
    // the binary so the loader finds the backends and rpath=$ORIGIN resolves.
    // std::fs::copy follows symlinks, so versioned SONAME files become real
    // files — correct for both NEEDED resolution and dlopen.
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    // OUT_DIR = target/<profile>/build/<crate>-<hash>/out  ->  target/<profile>/
    let profile_dir = out_dir
        .parent()
        .and_then(|p| p.parent())
        .and_then(|p| p.parent())
        .expect("OUT_DIR should be under target/<profile>/build/<crate>-<hash>/out");

    let copy_so = |src_dir: &std::path::Path| {
        if !src_dir.exists() {
            return;
        }
        for entry in std::fs::read_dir(src_dir).into_iter().flatten() {
            let path = match entry.ok() { Some(e) => e.path(), None => continue };
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            if !name.contains(".so") {
                continue;
            }
            let dst_path = profile_dir.join(name);
            // Only re-copy if missing/stale to keep incremental builds cheap.
            let needs_copy = match (std::fs::metadata(&dst_path), std::fs::metadata(&path)) {
                (Ok(dst_meta), Ok(src_meta)) => {
                    src_meta.modified().ok() > dst_meta.modified().ok()
                        || src_meta.len() != dst_meta.len()
                }
                (_, _) => true,
            };
            if needs_copy {
                if let Err(e) = std::fs::copy(&path, &dst_path) {
                    println!("cargo:warning=failed to copy {}: {}", path.display(), e);
                }
            }
        }
    };
    copy_so(&lib_dir);
    copy_so(&bin_dir);

    link_runtime_libs(openmp);
}

fn link_runtime_libs(openmp: bool) {
    // C++ runtime (libllama.so / the static archives are C++).
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
        if openmp {
            println!("cargo:rustc-link-lib=gomp"); // OpenMP
        }
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=msvcrt");
    }
}

fn configure_cuda_linking() {
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_ROOT"))
        .unwrap_or_else(|_| {
            if cfg!(target_os = "windows") {
                "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0".to_string()
            } else {
                "/usr/local/cuda".to_string()
            }
        });

    let cuda_lib_path = if cfg!(target_os = "windows") {
        format!("{}\\lib\\x64", cuda_path)
    } else {
        format!("{}/lib64", cuda_path)
    };

    println!("cargo:rustc-link-search=native={}", cuda_lib_path);
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=curand");

    // Check CUDA version
    if let Ok(output) = Command::new("nvcc").args(&["--version"]).output() {
        let version_str = String::from_utf8_lossy(&output.stdout);
        if version_str.contains("release 12") {
            println!("cargo:rustc-cfg=cuda_version=\"12\"");
        } else if version_str.contains("release 11") {
            println!("cargo:rustc-cfg=cuda_version=\"11\"");
        }
    }
}

fn configure_rocm_linking() {
    let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());

    println!("cargo:rustc-link-search=native={}/lib", rocm_path);
    println!("cargo:rustc-link-lib=hipblas");
    println!("cargo:rustc-link-lib=rocblas");
    println!("cargo:rustc-link-lib=amdhip64");
}

fn configure_opencl_linking() {
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=OpenCL");
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=framework=OpenCL");
    } else {
        if pkg_config::probe_library("OpenCL").is_ok() {
            println!("cargo:rustc-link-lib=OpenCL");
        } else {
            println!(
                "cargo:warning=OpenCL not found. Install opencl-headers and ocl-icd-opencl-dev"
            );
        }
    }

    // CLBlast for improved OpenCL performance
    if pkg_config::probe_library("clblast").is_ok() {
        println!("cargo:rustc-link-lib=clblast");
    }
}

// Print helpful error messages for missing dependencies
fn print_dependency_errors() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();

    match target_os.as_str() {
        "windows" => {
            if !command_exists("cl") && !command_exists("gcc") {
                println!("cargo:warning=No C++ compiler found. Install Visual Studio Build Tools or MinGW.");
            }
            if !command_exists("cmake") {
                println!("cargo:warning=CMake not found. Install CMake and add it to PATH.");
            }
        }
        "macos" => {
            if !command_exists("clang") {
                println!(
                    "cargo:warning=Xcode command line tools not found. Run: xcode-select --install"
                );
            }
            if !command_exists("cmake") {
                println!("cargo:warning=CMake not found. Install with: brew install cmake");
            }
        }
        "linux" => {
            if !command_exists("gcc") && !command_exists("clang") {
                println!("cargo:warning=No C++ compiler found. Install build-essential or clang.");
            }
            if !command_exists("cmake") {
                println!("cargo:warning=CMake not found. Install with your package manager.");
            }
            if !command_exists("pkg-config") {
                println!("cargo:warning=pkg-config not found. Install with your package manager.");
            }
        }
        _ => {}
    }
}

// Helper function to check if a command exists
fn command_exists(command: &str) -> bool {
    Command::new(command)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn generate_bindings(llama_cpp_path: &PathBuf, _build_path: &PathBuf) {
    let include_path = llama_cpp_path.join("include");
    let ggml_include_path = llama_cpp_path.join("ggml").join("include");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include_path.display()))
        .clang_arg(format!("-I{}", ggml_include_path.display()))
        .clang_arg(format!("-I{}/ggml/src", llama_cpp_path.display()))
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // Blocklist problematic types
        .blocklist_type("max_align_t")
        .blocklist_type("__off_t")
        .blocklist_type("__off64_t")
        .blocklist_type("_IO_lock_t")
        // Allow specific functions
        .allowlist_function("llama_.*")
        .allowlist_function("ggml_.*")
        // Allow specific types
        .allowlist_type("llama_.*")
        .allowlist_type("ggml_.*")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
