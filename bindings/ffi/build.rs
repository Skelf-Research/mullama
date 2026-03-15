//! Build script for mullama-ffi
//!
//! Generates C header file using cbindgen.

use std::env;
use std::path::PathBuf;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = PathBuf::from(&crate_dir).join("include");

    // Create include directory if it doesn't exist
    std::fs::create_dir_all(&out_dir).ok();

    // Generate C header using cbindgen
    let config = cbindgen::Config::from_file("cbindgen.toml")
        .unwrap_or_else(|_| cbindgen::Config::default());

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(out_dir.join("mullama.h"));
            println!("cargo:rerun-if-changed=src/lib.rs");
            println!("cargo:rerun-if-changed=src/model.rs");
            println!("cargo:rerun-if-changed=src/context.rs");
            println!("cargo:rerun-if-changed=src/sampler.rs");
            println!("cargo:rerun-if-changed=src/streaming.rs");
            println!("cargo:rerun-if-changed=src/embedding.rs");
            println!("cargo:rerun-if-changed=src/error.rs");
            println!("cargo:rerun-if-changed=src/handle.rs");
            println!("cargo:rerun-if-changed=cbindgen.toml");
        }
        Err(e) => {
            eprintln!("Warning: Failed to generate C header: {}", e);
            // Don't fail the build - the header can be regenerated manually
        }
    }
}
