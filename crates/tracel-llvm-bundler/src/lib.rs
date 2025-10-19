//! Minimal stub of the `tracel-llvm-bundler` crate used during builds that
//! operate without the full LLVM toolchain present.

pub mod config {
    /// No-op placeholder mirroring the upstream API.
    pub fn set_homebrew_library_path() -> Result<(), Box<dyn std::error::Error>> {
        Ok(())
    }

    /// Return a fixed LLVM major version for stub environments.
    pub fn init() -> Result<u32, Box<dyn std::error::Error>> {
        Ok(20)
    }

    /// Return a synthetic LLVM version string compatible with the expected major version.
    pub fn get_version(
        _prefix: Option<&std::ffi::OsString>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        Ok("20.0.0".to_string())
    }

    /// Return a placeholder include directory.
    pub fn get_includedir(
        _prefix: Option<&std::ffi::OsString>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        Ok("/tmp".to_string())
    }

    /// Return a placeholder library directory.
    pub fn get_libdir(
        _prefix: Option<&std::ffi::OsString>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        Ok("/tmp".to_string())
    }

    /// Provide an empty set of LLVM static libraries.
    pub fn get_libs(
        _prefix: Option<&std::ffi::OsString>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        Ok(Vec::new())
    }

    /// Provide an empty set of system libraries.
    pub fn get_system_libs(
        _prefix: Option<&std::ffi::OsString>,
    ) -> Result<Vec<String>, Box<dyn std::error::Error>> {
        Ok(Vec::new())
    }

    /// No system C++ library is required in the stub.
    pub fn get_system_libcpp() -> Option<String> {
        None
    }

    /// Return empty compiler flags.
    pub fn get_cxxflags(
        _prefix: Option<&std::ffi::OsString>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        Ok(String::new())
    }

    /// Return empty C compiler flags.
    pub fn get_cflags(
        _prefix: Option<&std::ffi::OsString>,
    ) -> Result<String, Box<dyn std::error::Error>> {
        Ok(String::new())
    }
}
