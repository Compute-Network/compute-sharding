use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let vendor_dir = manifest_dir.join("../../vendor/llama.cpp");
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("include/llama.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("src/llama-context.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("src/llama-graph.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("src/llama-graph.h").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("src/models/gemma.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("src/models/gemma2-iswa.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("src/models/gemma3.cpp").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        vendor_dir.join("src/models/gemma4-iswa.cpp").display()
    );

    let mut config = cmake::Config::new(&vendor_dir);
    config
        .define("BUILD_SHARED_LIBS", "ON")
        .define("LLAMA_BUILD_COMMON", "OFF")
        .define("LLAMA_BUILD_TESTS", "OFF")
        .define("LLAMA_BUILD_EXAMPLES", "OFF")
        .define("LLAMA_BUILD_SERVER", "OFF")
        .define("LLAMA_BUILD_TOOLS", "OFF")
        .define("LLAMA_TOOLS_INSTALL", "OFF")
        .define("LLAMA_TESTS_INSTALL", "OFF");
    if target_os == "macos" {
        config
            .define("GGML_METAL", "ON")
            .define("GGML_METAL_EMBED_LIBRARY", "ON");
        // Apple M-series hardware does not implement SVE/SME; the ggml ARM
        // feature probe uses check_cxx_source_runs, and running an SVE binary
        // on non-SVE hardware can wedge in kernel wait (process state UE).
        // Pre-seed the CMake cache to skip those run-time probes.
        config
            .define("GGML_MACHINE_SUPPORTS_sve", "OFF")
            .define("GGML_MACHINE_SUPPORTS_nosve", "ON")
            .define("GGML_MACHINE_SUPPORTS_sme", "OFF")
            .define("GGML_MACHINE_SUPPORTS_nosme", "ON");
    } else {
        config
            .define("GGML_METAL", "OFF")
            .define("GGML_METAL_EMBED_LIBRARY", "OFF");
    }

    let dst = config.build();

    let lib_dir = if target_os == "windows" {
        dst.join("bin")
    } else {
        dst.join("lib")
    };
    println!(
        "cargo:rustc-env=LLAMA_STAGE_VENDOR_LIB_DIR={}",
        lib_dir.display()
    );
}
