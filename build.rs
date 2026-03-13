use std::env;
use std::path::PathBuf;

fn main() {
    // Tell cargo to invalidate the built crate whenever wrapper files change
    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=build.rs");
    
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

fn build_llama_cpp(llama_cpp_path: &PathBuf) -> PathBuf {
    // This is a simplified build - in a real implementation, you'd want more features
    let dst = cmake::Config::new(llama_cpp_path)
        .define("LLAMA_NATIVE", "OFF")  // Disable native optimizations for broader compatibility
        .define("LLAMA_AVX", "ON")
        .define("LLAMA_AVX2", "ON")
        .define("LLAMA_AVX512", "OFF")  // Disable by default for compatibility
        .define("LLAMA_FMA", "ON")
        .define("LLAMA_F16C", "ON")
        .define("LLAMA_CUDA", "OFF")    // Disable CUDA by default
        .define("LLAMA_METAL", "OFF")   // Disable Metal by default
        .define("LLAMA_HIPBLAS", "OFF") // Disable ROCm by default
        .define("LLAMA_OPENMP", "ON")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .profile("Release")
        .build();
    
    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=dylib=llama");
    println!("cargo:rustc-link-lib=dylib=ggml");
    
    // Link to system libraries
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
    
    #[cfg(target_family = "unix")]
    {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    
    #[cfg(target_family = "windows")]
    {
        println!("cargo:rustc-link-lib=dylib=cublas");
        println!("cargo:rustc-link-lib=dylib=cublasLt");
    }
    
    dst
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