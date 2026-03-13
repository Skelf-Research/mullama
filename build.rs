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

    // General optimizations
    cmake_config.define("LLAMA_NATIVE", "ON");
    cmake_config.define("LLAMA_LTO", "ON");
    cmake_config.define("LLAMA_AVX", "ON");
    cmake_config.define("LLAMA_AVX2", "ON");
    cmake_config.define("LLAMA_FMA", "ON");
    cmake_config.define("LLAMA_F16C", "ON");
    cmake_config.define("LLAMA_OPENMP", "ON");

    // Build configuration
    cmake_config.define("LLAMA_BUILD_TESTS", "OFF");
    cmake_config.define("LLAMA_BUILD_EXAMPLES", "OFF");
    cmake_config.define("BUILD_SHARED_LIBS", "OFF");
    cmake_config.define("LLAMA_STATIC", "ON");

    let dst = cmake_config.build();

    // Link the built library
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=llama");

    // Platform-specific library linking - link all ggml components
    if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=static=ggml_static");
    } else {
        // Modern llama.cpp splits ggml into multiple libraries
        println!("cargo:rustc-link-lib=static=ggml");
        println!("cargo:rustc-link-lib=static=ggml-base");
        println!("cargo:rustc-link-lib=static=ggml-cpu");
    }

    // Link standard libraries
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
        println!("cargo:rustc-link-lib=gomp"); // OpenMP
    } else if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "windows") {
        println!("cargo:rustc-link-lib=msvcrt");
    }

    dst
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
