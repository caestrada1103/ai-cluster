//! Build script for compiling protobuf files

use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR")?);
    
    // Configure tonic build
    tonic_build::configure()
        .build_client(true)
        .build_server(true)
        .emit_rerun_if_changed(true)
        .file_descriptor_set_path(out_dir.join("cluster_descriptor.bin"))
        .compile(&["../proto/cluster.proto"], &["../proto"])?;
    
    // Rerun if protobuf changes
    println!("cargo:rerun-if-changed=../proto/cluster.proto");
    
    // Detect GPU backend
    #[cfg(feature = "hip")]
    {
        println!("cargo:rustc-cfg=feature=\"hip\"");
        println!("cargo:rerun-if-env-changed=ROCM_PATH");
        println!("cargo:rerun-if-env-changed=HIP_PATH");
        
        // Check for ROCm installation
        let rocm_path = env::var("ROCM_PATH").unwrap_or_else(|_| "/opt/rocm".to_string());
        let hip_path = format!("{}/hip", rocm_path);
        
        if std::path::Path::new(&rocm_path).exists() {
            println!("cargo:rustc-link-search=native={}/lib", rocm_path);
            println!("cargo:rustc-link-search=native={}/lib", hip_path);
            println!("cargo:rustc-link-lib=dylib=amdhip64");
            println!("cargo:rustc-link-lib=dylib=rccl");
            println!("cargo:rustc-flags=-l dylib=stdc++");
        }
    }
    
    #[cfg(feature = "cuda")]
    {
        println!("cargo:rustc-cfg=feature=\"cuda\"");
        println!("cargo:rerun-if-env-changed=CUDA_PATH");
        
        // Check for CUDA installation
        let cuda_path = env::var("CUDA_PATH").unwrap_or_else(|_| "/usr/local/cuda".to_string());
        
        if std::path::Path::new(&cuda_path).exists() {
            println!("cargo:rustc-link-search=native={}/lib64", cuda_path);
            println!("cargo:rustc-link-search=native={}/lib", cuda_path);
            println!("cargo:rustc-link-lib=dylib=cudart");
            println!("cargo:rustc-link-lib=dylib=nccl");
            println!("cargo:rustc-link-lib=dylib=cublas");
        }
    }
    
    Ok(())
}