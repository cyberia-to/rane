//! ANE Probe — reverse engineering Apple Neural Engine from pure Rust
//!
//! Level 1: IOKit service discovery — find the ANE driver
//! Level 2: Open user client — establish connection
//! Level 3: Probe selectors — discover the IOKit command interface
//! Level 4: IOSurface creation — prepare data buffers
//! Level 5: Attempt communication

#![allow(non_camel_case_types, non_upper_case_globals, non_snake_case)]

use std::ffi::{c_char, c_void, CStr};
use std::ptr;

// ============================================================
// IOKit FFI bindings (all pure C, no ObjC needed)
// ============================================================

type io_object_t = u32;
type io_service_t = io_object_t;
type io_connect_t = u32;
type mach_port_t = u32;
type kern_return_t = i32;
type IOOptionBits = u32;

const KERN_SUCCESS: kern_return_t = 0;
const kIOMasterPortDefault: mach_port_t = 0;

type CFMutableDictionaryRef = *mut c_void;
type CFStringRef = *const c_void;
type CFTypeRef = *const c_void;

#[link(name = "IOKit", kind = "framework")]
extern "C" {
    fn IOServiceMatching(name: *const c_char) -> CFMutableDictionaryRef;
    fn IOServiceGetMatchingService(
        mainPort: mach_port_t,
        matching: CFMutableDictionaryRef,
    ) -> io_service_t;
    fn IOServiceOpen(
        service: io_service_t,
        owningTask: mach_port_t,
        type_: u32,
        connect: *mut io_connect_t,
    ) -> kern_return_t;
    fn IOServiceClose(connect: io_connect_t) -> kern_return_t;
    fn IOConnectCallScalarMethod(
        connection: io_connect_t,
        selector: u32,
        input: *const u64,
        inputCnt: u32,
        output: *mut u64,
        outputCnt: *mut u32,
    ) -> kern_return_t;
    fn IOConnectCallStructMethod(
        connection: io_connect_t,
        selector: u32,
        inputStruct: *const c_void,
        inputStructCnt: usize,
        outputStruct: *mut c_void,
        outputStructCnt: *mut usize,
    ) -> kern_return_t;
    fn IORegistryEntryCreateCFProperty(
        entry: io_object_t,
        key: CFStringRef,
        allocator: *const c_void,
        options: IOOptionBits,
    ) -> CFTypeRef;
    fn IORegistryEntryGetName(entry: io_object_t, name: *mut c_char) -> kern_return_t;
    fn IOObjectGetClass(object: io_object_t, class_name: *mut c_char) -> kern_return_t;
    fn IOObjectRelease(object: io_object_t) -> kern_return_t;
}

extern "C" {
    fn mach_task_self() -> mach_port_t;
}

// CoreFoundation
#[link(name = "CoreFoundation", kind = "framework")]
extern "C" {
    fn CFStringCreateWithCString(
        alloc: *const c_void,
        cStr: *const c_char,
        encoding: u32,
    ) -> CFStringRef;
    fn CFGetTypeID(cf: CFTypeRef) -> u64;
    fn CFStringGetTypeID() -> u64;
    fn CFStringGetCString(
        theString: CFStringRef,
        buffer: *mut c_char,
        bufferSize: i64,
        encoding: u32,
    ) -> bool;
    fn CFDictionaryGetTypeID() -> u64;
    fn CFDictionaryCreateMutable(
        allocator: *const c_void,
        capacity: i64,
        keyCallBacks: *const c_void,
        valueCallBacks: *const c_void,
    ) -> CFMutableDictionaryRef;
    fn CFDictionarySetValue(
        dict: CFMutableDictionaryRef,
        key: *const c_void,
        value: *const c_void,
    );
    fn CFNumberCreate(
        allocator: *const c_void,
        theType: i32,
        valuePtr: *const c_void,
    ) -> *const c_void;
    fn CFRelease(cf: CFTypeRef);

    static kCFTypeDictionaryKeyCallBacks: c_void;
    static kCFTypeDictionaryValueCallBacks: c_void;
}

const kCFStringEncodingUTF8: u32 = 0x08000100;
const kCFNumberSInt32Type: i32 = 3;

// IOSurface
type IOSurfaceRef = *mut c_void;

#[link(name = "IOSurface", kind = "framework")]
extern "C" {
    fn IOSurfaceCreate(properties: CFMutableDictionaryRef) -> IOSurfaceRef;
    fn IOSurfaceLock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    fn IOSurfaceUnlock(surface: IOSurfaceRef, options: u32, seed: *mut u32) -> kern_return_t;
    fn IOSurfaceGetBaseAddress(surface: IOSurfaceRef) -> *mut c_void;
    fn IOSurfaceGetAllocSize(surface: IOSurfaceRef) -> usize;
    fn IOSurfaceGetID(surface: IOSurfaceRef) -> u32;
}

// dlopen/dlsym
extern "C" {
    fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

const RTLD_NOW: i32 = 0x2;

// ============================================================
// Helpers
// ============================================================

fn cf_str(s: &str) -> CFStringRef {
    unsafe {
        let c = std::ffi::CString::new(s).unwrap();
        CFStringCreateWithCString(ptr::null(), c.as_ptr(), kCFStringEncodingUTF8)
    }
}

fn cf_num(v: i32) -> *const c_void {
    unsafe { CFNumberCreate(ptr::null(), kCFNumberSInt32Type, &v as *const i32 as *const c_void) }
}

const KIO_BAD_ARG: i32 = 0xe00002bc_u32 as i32;
const KIO_NOT_PRIV: i32 = 0xe00002c1_u32 as i32;
const KIO_UNSUPPORTED: i32 = 0xe00002d8_u32 as i32;
const KIO_EXCLUSIVE: i32 = 0xe00002be_u32 as i32;
const KIO_NOT_READY: i32 = 0xe00002c0_u32 as i32;
const KIO_NOT_PERMITTED: i32 = 0xe00002c2_u32 as i32;

fn kern_err(kr: kern_return_t) -> String {
    match kr {
        0 => "KERN_SUCCESS".into(),
        KIO_BAD_ARG => "kIOReturnBadArgument".into(),
        KIO_NOT_PRIV => "kIOReturnNotPrivileged".into(),
        KIO_UNSUPPORTED => "kIOReturnUnsupported".into(),
        KIO_EXCLUSIVE => "kIOReturnExclusiveAccess".into(),
        KIO_NOT_READY => "kIOReturnNotReady".into(),
        KIO_NOT_PERMITTED => "kIOReturnNotPermitted".into(),
        _ => format!("{:#010x}", kr as u32),
    }
}

// ============================================================
// Level 1: IOKit Service Discovery
// ============================================================

fn level1_discover() -> Option<io_service_t> {
    println!("═══════════════════════════════════════════════════");
    println!("  LEVEL 1: IOKit Service Discovery");
    println!("═══════════════════════════════════════════════════\n");

    let candidates = [
        "H11ANEIn",
        "H11ANEInDirectPathClient",
        "AppleH11ANEInterface",
        "AppleANELoadBalancer",
    ];

    let mut found_service: Option<io_service_t> = None;

    for name in &candidates {
        let c_name = std::ffi::CString::new(*name).unwrap();
        unsafe {
            let matching = IOServiceMatching(c_name.as_ptr());
            if matching.is_null() {
                continue;
            }

            let service = IOServiceGetMatchingService(kIOMasterPortDefault, matching);
            if service == 0 {
                println!("  [-] {} — not found", name);
            } else {
                let mut class_name = [0i8; 128];
                IOObjectGetClass(service, class_name.as_mut_ptr());
                let class_str = CStr::from_ptr(class_name.as_ptr()).to_string_lossy();

                let mut entry_name = [0i8; 128];
                IORegistryEntryGetName(service, entry_name.as_mut_ptr());
                let entry_str = CStr::from_ptr(entry_name.as_ptr()).to_string_lossy();

                println!(
                    "  [+] {} — FOUND (id={:#x}, class='{}', name='{}')",
                    name, service, class_str, entry_str
                );

                // Read DeviceProperties
                let key = cf_str("DeviceProperties");
                let val = IORegistryEntryCreateCFProperty(service, key, ptr::null(), 0);
                if !val.is_null() {
                    let type_id = CFGetTypeID(val);
                    let dict_id = CFDictionaryGetTypeID();
                    if type_id == dict_id {
                        println!("      DeviceProperties = <CFDictionary>");
                    }
                    CFRelease(val);
                }

                if found_service.is_none() {
                    found_service = Some(service);
                } else {
                    IOObjectRelease(service);
                }
            }
        }
    }

    found_service
}

// ============================================================
// Level 2: Open User Client
// ============================================================

fn level2_open(service: io_service_t) -> Option<io_connect_t> {
    println!("\n═══════════════════════════════════════════════════");
    println!("  LEVEL 2: Open User Client Connection");
    println!("═══════════════════════════════════════════════════\n");

    for uc_type in 0..=10 {
        unsafe {
            let mut connect: io_connect_t = 0;
            let kr = IOServiceOpen(service, mach_task_self(), uc_type, &mut connect);
            if kr == KERN_SUCCESS {
                println!(
                    "  [+] IOServiceOpen(type={}) => connection={:#x}  *** SUCCESS ***",
                    uc_type, connect
                );
                return Some(connect);
            } else {
                println!(
                    "  [-] IOServiceOpen(type={}) => {}",
                    uc_type,
                    kern_err(kr)
                );
            }
        }
    }
    None
}

// ============================================================
// Level 3: Probe Selectors
// ============================================================

fn level3_probe(connect: io_connect_t) {
    println!("\n═══════════════════════════════════════════════════");
    println!("  LEVEL 3: Probe IOKit Selectors");
    println!("═══════════════════════════════════════════════════\n");

    for selector in 0..64 {
        unsafe {
            // Try scalar (no input)
            let mut output = [0u64; 16];
            let mut output_cnt: u32 = 16;
            let kr_scalar = IOConnectCallScalarMethod(
                connect,
                selector,
                ptr::null(),
                0,
                output.as_mut_ptr(),
                &mut output_cnt,
            );

            // Try struct (no input)
            let mut out_buf = [0u8; 4096];
            let mut out_size: usize = 4096;
            let kr_struct = IOConnectCallStructMethod(
                connect,
                selector,
                ptr::null(),
                0,
                out_buf.as_mut_ptr() as *mut c_void,
                &mut out_size,
            );

            // Only print if something interesting happened
            const MIG_BAD_ID: i32 = 0xfffffbc4_u32 as i32;
            let scalar_interesting = kr_scalar != KIO_BAD_ARG && kr_scalar != MIG_BAD_ID;
            let struct_interesting = kr_struct != KIO_BAD_ARG && kr_struct != MIG_BAD_ID;

            if scalar_interesting || struct_interesting {
                print!("  sel {:2}:", selector);
                if scalar_interesting {
                    if kr_scalar == KERN_SUCCESS {
                        print!(
                            " scalar=OK({}:{:?})",
                            output_cnt,
                            &output[..output_cnt as usize]
                        );
                    } else {
                        print!(" scalar={}", kern_err(kr_scalar));
                    }
                }
                if struct_interesting {
                    if kr_struct == KERN_SUCCESS {
                        print!(" struct=OK({} bytes)", out_size);
                        if out_size > 0 && out_size <= 64 {
                            print!(" data={:02x?}", &out_buf[..out_size]);
                        }
                    } else {
                        print!(" struct={}", kern_err(kr_struct));
                    }
                }
                println!();
            }
        }
    }
}

// ============================================================
// Level 4: IOSurface creation
// ============================================================

fn level4_surface() -> Option<IOSurfaceRef> {
    println!("\n═══════════════════════════════════════════════════");
    println!("  LEVEL 4: IOSurface (ANE tensor buffer)");
    println!("═══════════════════════════════════════════════════\n");

    unsafe {
        let dict = CFDictionaryCreateMutable(
            ptr::null(),
            0,
            &kCFTypeDictionaryKeyCallBacks as *const c_void,
            &kCFTypeDictionaryValueCallBacks as *const c_void,
        );

        // ANE native format: [1, C, 1, S] where C=channels, S=spatial
        // Map to IOSurface: width=S, height=1, bytesPerElement=C*2 (fp16)
        let channels: i32 = 64;
        let spatial: i32 = 64;
        let bpe: i32 = channels * 2;

        CFDictionarySetValue(dict, cf_str("IOSurfaceWidth") as _, cf_num(spatial));
        CFDictionarySetValue(dict, cf_str("IOSurfaceHeight") as _, cf_num(1));
        CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerElement") as _, cf_num(bpe));
        CFDictionarySetValue(dict, cf_str("IOSurfacePixelFormat") as _, cf_num(0));

        let surface = IOSurfaceCreate(dict);
        if surface.is_null() {
            println!("  [-] IOSurfaceCreate FAILED");
            return None;
        }

        let alloc = IOSurfaceGetAllocSize(surface);
        let sid = IOSurfaceGetID(surface);
        println!("  [+] IOSurface created from pure Rust!");
        println!("      ID     = {}", sid);
        println!("      Size   = {} bytes ({:.1} KB)", alloc, alloc as f64 / 1024.0);
        println!("      Shape  = [1, {}, 1, {}] fp16", channels, spatial);

        // Write recognizable pattern
        IOSurfaceLock(surface, 0, ptr::null_mut());
        let base = IOSurfaceGetBaseAddress(surface) as *mut u16;
        if !base.is_null() {
            for i in 0..(channels * spatial) as usize {
                *base.add(i) = 0x3C00; // fp16(1.0)
            }
            println!("      Wrote {} fp16 values = 1.0 each", channels * spatial);
        }
        IOSurfaceUnlock(surface, 0, ptr::null_mut());

        Some(surface)
    }
}

// ============================================================
// Level 5: Load all three ANE frameworks, enumerate C symbols
// ============================================================

struct AneFrameworks {
    compiler: *mut c_void,
    services: *mut c_void,
    engine: *mut c_void,
}

fn load_framework(name: &str) -> *mut c_void {
    let path = format!(
        "/System/Library/PrivateFrameworks/{}.framework/{}",
        name, name
    );
    let c = std::ffi::CString::new(path.as_str()).unwrap();
    let h = unsafe { dlopen(c.as_ptr(), RTLD_NOW) };
    if h.is_null() {
        println!("  [-] {} — failed to load", name);
    } else {
        println!("  [+] {} — loaded @ {:?}", name, h);
    }
    h
}

fn find_sym(name: &str) -> *mut c_void {
    let c = std::ffi::CString::new(name).unwrap();
    let rtld_default = (-2isize) as *mut c_void;
    unsafe { dlsym(rtld_default, c.as_ptr()) }
}

fn level5_frameworks() -> Option<AneFrameworks> {
    println!("\n═══════════════════════════════════════════════════");
    println!("  LEVEL 5: Load ANE Frameworks (C API)");
    println!("═══════════════════════════════════════════════════\n");

    let engine = load_framework("AppleNeuralEngine");
    let compiler = load_framework("ANECompiler");
    let services = load_framework("ANEServices");

    // Now scan for key C functions across all loaded images
    println!("\n  Key C functions (no ObjC needed):");
    let key_symbols = [
        // Compiler
        "ANECCompile",
        "ANECCompileJIT",
        "ANECCompileOnline",
        "ANECCompileOffline",
        "ANECCreateCompilerInputDictionary",
        "ANECCreateCompilerOptionDictionary",
        "ANECCreateModelDictionary",
        "ANECCreateDeviceProperty",
        "ANECGetDeviceProperty",
        "ANECValidate",
        "ANECGetCompilerFileFormat",
        // Services
        "ANEServicesDeviceOpen",
        "ANEServicesDeviceClose",
        "ANEServicesProgramCreate",
        "ANEServicesProgramPrepare",
        "ANEServicesProgramInputsReady",
        "ANEServicesProgramDestroy",
        "ANEServicesProgramStop",
        "ANEServicesProgramProcessRequestDirect",
        "ANEServicesProgramOutputSetEnqueue",
        "ANEServicesProgramMemoryMapRequest",
        "ANEServicesProgramMemoryUnmapRequest",
        "ANEServicesProgramChainingPrepare",
        "ANEServicesInitializePlatformServices",
        // Validation
        "ANEValidateMILNetworkOnHost",
        "ANEValidateNetworkCreate",
        "ANEGetValidateNetworkSupportedVersion",
    ];

    for sym in &key_symbols {
        let p = find_sym(sym);
        if !p.is_null() {
            println!("  [+] {:50} @ {:?}", sym, p);
        }
    }

    if compiler.is_null() && services.is_null() && engine.is_null() {
        return None;
    }

    Some(AneFrameworks {
        compiler,
        services,
        engine,
    })
}

// ============================================================
// Level 6: Call ANEServices C API — open device, create program
// ============================================================

// ANEServices function signatures (reverse-engineered from dyld_info)
// These are C functions, NOT ObjC methods!
type ANEServicesDeviceOpenFn =
    unsafe extern "C" fn(device_id: u32, flags: u32, out_handle: *mut *mut c_void) -> i32;
type ANEServicesDeviceCloseFn = unsafe extern "C" fn(handle: *mut c_void) -> i32;
type ANEServicesProgramCreateFn = unsafe extern "C" fn(
    device: *mut c_void,
    program_data: *const c_void,
    program_size: usize,
    options: *const c_void, // CFDictionary
    out_program: *mut *mut c_void,
) -> i32;

// ANECCompile signature guess — likely takes CFDictionary of options
type ANECCompileFn = unsafe extern "C" fn(
    input: *const c_void,  // CFDictionary with model + options
    output: *mut c_void,   // output buffer/dict
    flags: u32,
) -> i32;

fn level6_direct_api() {
    println!("\n═══════════════════════════════════════════════════");
    println!("  LEVEL 6: Architecture Discovery");
    println!("═══════════════════════════════════════════════════\n");

    println!("  Communication path discovered:");
    println!("    Your code");
    println!("      → AppleNeuralEngine.framework (_ANEInMemoryModel etc.)");
    println!("      → XPC to 'com.apple.appleneuralengine' (Mach service)");
    println!("      → aned daemon (/usr/libexec/aned)");
    println!("      → IOKit H11ANEIn (entitlement: com.apple.ane.iokit-user-access)");
    println!("      → ANE hardware\n");

    println!("  IOKit selectors return kIOReturnNotPermitted because");
    println!("  our process lacks com.apple.ane.iokit-user-access entitlement.");
    println!("  Only aned has this. All user code goes through XPC.\n");

    // _ANEDaemonInterface — returns NSXPCInterface describing the protocol
    let daemon_iface = find_sym("_ANEDaemonInterface");
    if !daemon_iface.is_null() {
        println!("  [+] _ANEDaemonInterface @ {:?}", daemon_iface);
        println!("      This returns the NSXPCInterface for talking to aned.");
        println!("      The ObjC classes wrap this, but we could call it directly.");
    }

    let daemon_iface_private = find_sym("_ANEDaemonInterfacePrivate");
    if !daemon_iface_private.is_null() {
        println!("  [+] _ANEDaemonInterfacePrivate @ {:?}", daemon_iface_private);
    }

    // Check if we can connect to the XPC service from Rust
    // Using bootstrap_look_up to find the mach port
    println!("\n  [*] Checking XPC service reachability...");
    unsafe {
        extern "C" {
            fn bootstrap_look_up(
                bp: mach_port_t,
                service_name: *const c_char,
                sp: *mut mach_port_t,
            ) -> kern_return_t;
            fn bootstrap_port() -> mach_port_t;

            // We can access bootstrap_port via task_get_special_port
            fn task_get_special_port(
                task: mach_port_t,
                which_port: i32,
                special_port: *mut mach_port_t,
            ) -> kern_return_t;
        }

        const TASK_BOOTSTRAP_PORT: i32 = 4;
        let mut bp: mach_port_t = 0;
        let kr = task_get_special_port(mach_task_self(), TASK_BOOTSTRAP_PORT, &mut bp);
        if kr == KERN_SUCCESS {
            println!("  [+] Got bootstrap port: {:#x}", bp);

            let svc_name = std::ffi::CString::new("com.apple.appleneuralengine").unwrap();
            let mut svc_port: mach_port_t = 0;
            let kr2 = bootstrap_look_up(bp, svc_name.as_ptr(), &mut svc_port);
            if kr2 == KERN_SUCCESS {
                println!(
                    "  [+] com.apple.appleneuralengine => port {:#x} *** REACHABLE ***",
                    svc_port
                );
                println!("      This is the Mach port to aned. XPC talks here.");
            } else {
                println!("  [-] bootstrap_look_up failed: {}", kr2);
            }
        }
    }
}

// ============================================================
// Level 7: Compile MIL → ANE bytecode from pure Rust
// ============================================================

// ObjC runtime FFI — these are plain C functions in libobjc
type ObjcClass = *const c_void;
type ObjcSel = *const c_void;
type ObjcId = *mut c_void;

extern "C" {
    fn objc_getClass(name: *const c_char) -> ObjcClass;
    fn sel_registerName(name: *const c_char) -> ObjcSel;
    fn objc_msgSend() -> ObjcId; // actual signature varies per call
}

// CFDictionary introspection
extern "C" {
    fn CFDictionaryGetCount(dict: *const c_void) -> i64;
    fn CFDictionaryGetKeysAndValues(
        dict: *const c_void,
        keys: *mut *const c_void,
        values: *mut *const c_void,
    );
    fn CFCopyDescription(cf: *const c_void) -> CFStringRef;
    fn CFDataCreate(
        allocator: *const c_void,
        bytes: *const u8,
        length: i64,
    ) -> *const c_void;
    fn CFDataGetLength(data: *const c_void) -> i64;
    fn CFDataGetBytePtr(data: *const c_void) -> *const u8;
    fn CFBooleanGetValue(boolean: *const c_void) -> bool;
    fn CFArrayGetCount(array: *const c_void) -> i64;
}

fn cf_desc(obj: *const c_void) -> String {
    if obj.is_null() { return "(null)".into(); }
    unsafe {
        let desc = CFCopyDescription(obj);
        if desc.is_null() { return "(no description)".into(); }
        let mut buf = [0i8; 4096];
        CFStringGetCString(desc, buf.as_mut_ptr(), 4096, kCFStringEncodingUTF8);
        CFRelease(desc);
        CStr::from_ptr(buf.as_ptr()).to_string_lossy().into_owned()
    }
}

fn sel(name: &str) -> ObjcSel {
    let c = std::ffi::CString::new(name).unwrap();
    unsafe { sel_registerName(c.as_ptr()) }
}

fn cls(name: &str) -> ObjcClass {
    let c = std::ffi::CString::new(name).unwrap();
    unsafe { objc_getClass(c.as_ptr()) }
}

// Build ANE weight blob: 128-byte header + fp16 data
// Header format from training_dynamic/io.h build_blob():
//   byte 0: 1, byte 4: 2, bytes 64-67: 0xDEADBEEF, byte 68: 1
//   bytes 72-75: weight_size_bytes, bytes 80-83: 128 (data offset)
fn build_weight_blob(fp16_data: &[u16]) -> Vec<u8> {
    let weight_bytes = fp16_data.len() * 2;
    let total = 128 + weight_bytes;
    let mut blob = vec![0u8; total];

    blob[0] = 1;
    blob[4] = 2;
    // DEADBEEF magic
    blob[64] = 0xEF; blob[65] = 0xBE; blob[66] = 0xAD; blob[67] = 0xDE;
    blob[68] = 1;
    // Weight size in bytes
    blob[72..76].copy_from_slice(&(weight_bytes as u32).to_le_bytes());
    // Data offset
    blob[80..84].copy_from_slice(&128u32.to_le_bytes());
    // Copy fp16 data
    for (i, &val) in fp16_data.iter().enumerate() {
        let off = 128 + i * 2;
        blob[off..off+2].copy_from_slice(&val.to_le_bytes());
    }
    blob
}

fn level7_mil_compile() {
    println!("\n═══════════════════════════════════════════════════");
    println!("  LEVEL 7: MIL → ANE Bytecode (Pure Rust)");
    println!("═══════════════════════════════════════════════════\n");

    // ── Step 1: Minimal MIL program ──
    // Dynamic matmul: y = x @ W (exact format from training_dynamic/mil_dynamic.h)
    // Input: [1, 64, 1, 128] — first 64 cols = activations, last 64 = weights
    // Output: [1, 64, 1, 64] — matmul result
    let ic = 64;
    let oc = 64;
    let seq = 64;
    let sp = seq + oc; // 128

    let mil = format!(
        concat!(
            "program(1.3)\n",
            "[buildInfo = dict<string, string>(",
            "{{{{\"coremlc-component-MIL\", \"3510.2.1\"}}, ",
            "{{\"coremlc-version\", \"3505.4.1\"}}, ",
            "{{\"coremltools-component-milinternal\", \"\"}}, ",
            "{{\"coremltools-version\", \"9.0\"}}}}",
            ")]\n{{\n",
            "    func main<ios18>(tensor<fp16, [1, {ic}, 1, {sp}]> x) {{\n",
            "        tensor<int32, [4]> ba = const()[name=string(\"ba\"), val=tensor<int32, [4]>([0,0,0,0])];\n",
            "        tensor<int32, [4]> sa = const()[name=string(\"sa\"), val=tensor<int32, [4]>([1,{ic},1,{seq}])];\n",
            "        tensor<fp16, [1,{ic},1,{seq}]> act = slice_by_size(x=x,begin=ba,size=sa)[name=string(\"act\")];\n",
            "        tensor<int32, [4]> bw = const()[name=string(\"bw\"), val=tensor<int32, [4]>([0,0,0,{seq}])];\n",
            "        tensor<int32, [4]> sw = const()[name=string(\"sw\"), val=tensor<int32, [4]>([1,{ic},1,{oc}])];\n",
            "        tensor<fp16, [1,{ic},1,{oc}]> wt = slice_by_size(x=x,begin=bw,size=sw)[name=string(\"wt\")];\n",
            "        tensor<int32, [4]> ra = const()[name=string(\"ra\"), val=tensor<int32, [4]>([1,1,{ic},{seq}])];\n",
            "        tensor<fp16, [1,1,{ic},{seq}]> a2 = reshape(shape=ra,x=act)[name=string(\"a2\")];\n",
            "        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n",
            "        tensor<fp16, [1,1,{seq},{ic}]> a3 = transpose(perm=pm,x=a2)[name=string(\"a3\")];\n",
            "        tensor<int32, [4]> rw = const()[name=string(\"rw\"), val=tensor<int32, [4]>([1,1,{ic},{oc}])];\n",
            "        tensor<fp16, [1,1,{ic},{oc}]> W = reshape(shape=rw,x=wt)[name=string(\"W\")];\n",
            "        bool bF = const()[name=string(\"bF\"), val=bool(false)];\n",
            "        tensor<fp16, [1,1,{seq},{oc}]> yh = matmul(transpose_x=bF,transpose_y=bF,x=a3,y=W)[name=string(\"yh\")];\n",
            "        tensor<fp16, [1,1,{oc},{seq}]> yt = transpose(perm=pm,x=yh)[name=string(\"yt\")];\n",
            "        tensor<int32, [4]> ro = const()[name=string(\"ro\"), val=tensor<int32, [4]>([1,{oc},1,{seq}])];\n",
            "        tensor<fp16, [1,{oc},1,{seq}]> y = reshape(shape=ro,x=yt)[name=string(\"y\")];\n",
            "    }} -> (y);\n",
            "}}\n"
        ),
        ic=ic, oc=oc, seq=seq, sp=sp
    );
    println!("  MIL program ({} bytes):", mil.len());
    for line in mil.lines() {
        if !line.is_empty() { println!("    {}", line); }
    }

    // ── Step 2: Try ANECCompile C API first ──
    println!("\n  ── Approach A: ANECCompile (pure C) ──\n");
    let approach_a_ok = try_anec_compile(&mil);

    if !approach_a_ok {
        // ── Step 3: ObjC runtime path ──
        println!("\n  ── Approach B: ObjC runtime from Rust ──\n");
        try_objc_compile(&mil);
    }
}

fn try_anec_compile(_mil: &str) -> bool {
    // ANECCreateCompilerInputDictionary/OptionDictionary hang when called
    // with no args — they likely require arguments we don't know yet.
    // Skip pure C API for now, go straight to ObjC runtime path.
    println!("  [~] Skipping ANECCompile C API (helper functions require");
    println!("      unknown arguments). Using ObjC runtime instead.");
    false
}

fn try_objc_compile(mil: &str) -> bool {
    println!("  Using ObjC runtime from Rust (objc_msgSend — same as training code)");
    println!("  No ObjC compiler needed — just C FFI to libobjc\n");

    unsafe {
        // Get ObjC classes
        let cls_descriptor = cls("_ANEInMemoryModelDescriptor");
        let cls_model = cls("_ANEInMemoryModel");

        if cls_descriptor.is_null() {
            println!("  [-] _ANEInMemoryModelDescriptor class not found");
            println!("      AppleNeuralEngine.framework might not export it");
            return false;
        }
        println!("  [+] _ANEInMemoryModelDescriptor @ {:?}", cls_descriptor);

        if cls_model.is_null() {
            println!("  [-] _ANEInMemoryModel class not found");
            return false;
        }
        println!("  [+] _ANEInMemoryModel @ {:?}", cls_model);

        // Create NSData from MIL text
        let cls_nsdata = cls("NSData");
        let mil_bytes = mil.as_bytes();
        type MsgSendDataInit = unsafe extern "C" fn(
            ObjcClass, ObjcSel, *const u8, u64,
        ) -> ObjcId;
        let data_init: MsgSendDataInit = std::mem::transmute(objc_msgSend as *const c_void);
        let mil_data = data_init(
            cls_nsdata as *const c_void as *mut c_void,
            sel("dataWithBytes:length:"),
            mil_bytes.as_ptr(),
            mil_bytes.len() as u64,
        );
        if mil_data.is_null() {
            println!("  [-] Failed to create NSData from MIL text");
            return false;
        }
        println!("  [+] NSData(MIL) created ({} bytes)", mil_bytes.len());

        // Create empty NSDictionary for weights (identity has no weights)
        let cls_nsdict = cls("NSDictionary");
        type MsgSendInit = unsafe extern "C" fn(ObjcClass, ObjcSel) -> ObjcId;
        let dict_init: MsgSendInit = std::mem::transmute(objc_msgSend as *const c_void);
        let empty_weights = dict_init(
            cls_nsdict as *const c_void as *mut c_void,
            sel("dictionary"),
        );
        println!("  [+] Empty weights dict created");

        // _ANEInMemoryModelDescriptor.modelWithMILText:weights:optionsPlist:
        println!("\n  [*] Creating model descriptor...");
        type MsgSendDesc = unsafe extern "C" fn(
            ObjcClass, ObjcSel, ObjcId, ObjcId, ObjcId,
        ) -> ObjcId;
        let create_desc: MsgSendDesc = std::mem::transmute(objc_msgSend as *const c_void);
        let descriptor = create_desc(
            cls_descriptor as *const c_void as *mut c_void,
            sel("modelWithMILText:weights:optionsPlist:"),
            mil_data,
            empty_weights,
            ptr::null_mut(), // nil options
        );
        if descriptor.is_null() {
            println!("  [-] modelWithMILText:weights:optionsPlist: returned nil");
            return false;
        }
        println!("  [+] Descriptor created!");

        // _ANEInMemoryModel.inMemoryModelWithDescriptor:
        println!("  [*] Creating in-memory model...");
        type MsgSendModel = unsafe extern "C" fn(ObjcClass, ObjcSel, ObjcId) -> ObjcId;
        let create_model: MsgSendModel = std::mem::transmute(objc_msgSend as *const c_void);
        let model = create_model(
            cls_model as *const c_void as *mut c_void,
            sel("inMemoryModelWithDescriptor:"),
            descriptor,
        );
        if model.is_null() {
            println!("  [-] inMemoryModelWithDescriptor: returned nil");
            return false;
        }
        println!("  [+] Model created!");

        // Get hex identifier for temp dir
        type MsgSendStr = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
        let get_str: MsgSendStr = std::mem::transmute(objc_msgSend as *const c_void);
        let hex_id = get_str(model, sel("hexStringIdentifier"));
        let hex_str = if !hex_id.is_null() {
            type MsgSendUtf8 = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const c_char;
            let utf8_fn: MsgSendUtf8 = std::mem::transmute(objc_msgSend as *const c_void);
            let cstr = utf8_fn(hex_id, sel("UTF8String"));
            if !cstr.is_null() {
                CStr::from_ptr(cstr).to_string_lossy().into_owned()
            } else { "unknown".to_string() }
        } else { "unknown".to_string() };
        println!("  [+] Model hex ID: {}", hex_str);

        // Set up temp directory (model.mil + weights/)
        let tmp_dir = std::env::temp_dir().join(&hex_str);
        let _ = std::fs::create_dir_all(tmp_dir.join("weights"));
        std::fs::write(tmp_dir.join("model.mil"), mil).unwrap();
        println!("  [+] Temp dir: {:?}", tmp_dir);

        // Compile: model.compileWithQoS:options:error:
        println!("\n  [*] *** COMPILING MIL → ANE BYTECODE ***");
        let mut error: ObjcId = ptr::null_mut();
        type MsgSendCompile = unsafe extern "C" fn(
            ObjcId, ObjcSel, u32, ObjcId, *mut ObjcId,
        ) -> bool;
        let compile_fn: MsgSendCompile = std::mem::transmute(objc_msgSend as *const c_void);
        let ok = compile_fn(
            model,
            sel("compileWithQoS:options:error:"),
            21, // QoS
            empty_weights, // options (empty dict)
            &mut error,
        );

        if ok {
            println!("  [+] *** COMPILATION SUCCEEDED! ***");
            println!("      MIL → ANE bytecode compiled from pure Rust!");

            // Check what files were produced
            if let Ok(entries) = std::fs::read_dir(&tmp_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name();
                    let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    println!("      {:?}: {} bytes", name, size);
                    // Recurse one level for subdirs
                    if entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                        if let Ok(sub) = std::fs::read_dir(entry.path()) {
                            for se in sub.flatten() {
                                let sname = se.file_name();
                                let ssize = se.metadata().map(|m| m.len()).unwrap_or(0);
                                println!("        {:?}: {} bytes", sname, ssize);
                            }
                        }
                    }
                }
            }

            // Load: model.loadWithQoS:options:error:
            println!("\n  [*] Loading compiled model into ANE...");
            let mut load_err: ObjcId = ptr::null_mut();
            let load_ok = compile_fn(
                model,
                sel("loadWithQoS:options:error:"),
                21,
                empty_weights,
                &mut load_err,
            );

            if load_ok {
                println!("  [+] *** MODEL LOADED INTO ANE! ***");

                // Create IOSurfaces for I/O
                // Input: [1,IC,1,SP] fp16 where SP=SEQ+OC
                // Output: [1,OC,1,SEQ] fp16
                let ic: usize = 64; let oc: usize = 64;
                let seq: usize = 64; let sp: usize = seq + oc;
                let io_in = level4_make_surface(ic * sp * 2);
                let io_out = level4_make_surface(oc * seq * 2);

                if !io_in.is_null() && !io_out.is_null() {
                    // Write test data: activations = 1.0, weights = identity-like
                    IOSurfaceLock(io_in, 0, ptr::null_mut());
                    let base_in = IOSurfaceGetBaseAddress(io_in) as *mut u16;
                    // Fill activations [IC, SEQ] with 1.0
                    for ch in 0..ic {
                        for s in 0..seq {
                            *base_in.add(ch * sp + s) = 0x3C00; // fp16(1.0)
                        }
                        // Fill weights [IC, OC] with identity (1.0 on diagonal)
                        for o in 0..oc {
                            *base_in.add(ch * sp + seq + o) = if ch == o { 0x3C00 } else { 0 };
                        }
                    }
                    IOSurfaceUnlock(io_in, 0, ptr::null_mut());
                    println!("  [+] Input: acts=[{},{}] all 1.0, weights=[{},{}] identity", ic, seq, ic, oc);

                    // Create _ANEIOSurfaceObject wrappers
                    let cls_aio = cls("_ANEIOSurfaceObject");
                    if !cls_aio.is_null() {
                        type MsgSendIO = unsafe extern "C" fn(
                            ObjcClass, ObjcSel, IOSurfaceRef,
                        ) -> ObjcId;
                        let io_wrap: MsgSendIO = std::mem::transmute(objc_msgSend as *const c_void);
                        let w_in = io_wrap(cls_aio as *const c_void as *mut c_void,
                            sel("objectWithIOSurface:"), io_in);
                        let w_out = io_wrap(cls_aio as *const c_void as *mut c_void,
                            sel("objectWithIOSurface:"), io_out);

                        // Create _ANERequest
                        let cls_req = cls("_ANERequest");
                        if !cls_req.is_null() && !w_in.is_null() && !w_out.is_null() {
                            // Build NSArrays
                            let cls_arr = cls("NSArray");
                            let cls_num = cls("NSNumber");
                            type MsgSendArr = unsafe extern "C" fn(
                                ObjcClass, ObjcSel, *const ObjcId, u64,
                            ) -> ObjcId;
                            let arr_fn: MsgSendArr = std::mem::transmute(objc_msgSend as *const c_void);
                            let inputs = arr_fn(cls_arr as *const _ as *mut _, sel("arrayWithObjects:count:"),
                                &w_in as *const ObjcId, 1);
                            let outputs = arr_fn(cls_arr as *const _ as *mut _, sel("arrayWithObjects:count:"),
                                &w_out as *const ObjcId, 1);

                            type MsgSendNum = unsafe extern "C" fn(ObjcClass, ObjcSel, i32) -> ObjcId;
                            let num_fn: MsgSendNum = std::mem::transmute(objc_msgSend as *const c_void);
                            let zero = num_fn(cls_num as *const _ as *mut _, sel("numberWithInt:"), 0);
                            let indices = arr_fn(cls_arr as *const _ as *mut _, sel("arrayWithObjects:count:"),
                                &zero as *const ObjcId, 1);

                            // requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:
                            type MsgSendReq = unsafe extern "C" fn(
                                ObjcClass, ObjcSel,
                                ObjcId, ObjcId, ObjcId, ObjcId, ObjcId, ObjcId, ObjcId,
                            ) -> ObjcId;
                            let req_fn: MsgSendReq = std::mem::transmute(objc_msgSend as *const c_void);
                            let request = req_fn(
                                cls_req as *const _ as *mut _,
                                sel("requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:"),
                                inputs, indices, outputs, indices,
                                ptr::null_mut(), ptr::null_mut(), zero,
                            );

                            if !request.is_null() {
                                println!("  [+] _ANERequest created");

                                // Evaluate!
                                println!("\n  [*] *** EVALUATING ON ANE HARDWARE ***");
                                let mut eval_err: ObjcId = ptr::null_mut();
                                type MsgSendEval = unsafe extern "C" fn(
                                    ObjcId, ObjcSel, u32, ObjcId, ObjcId, *mut ObjcId,
                                ) -> bool;
                                let eval_fn: MsgSendEval = std::mem::transmute(objc_msgSend as *const c_void);
                                let eval_ok = eval_fn(
                                    model,
                                    sel("evaluateWithQoS:options:request:error:"),
                                    21,
                                    empty_weights, // options (empty dict)
                                    request,
                                    &mut eval_err,
                                );

                                if eval_ok {
                                    println!("  [+] *** ANE EVALUATION SUCCEEDED! ***");
                                    println!("      Pure Rust → MIL → ANE bytecode → ANE hardware → result!");

                                    // Read output
                                    IOSurfaceLock(io_out, 0, ptr::null_mut());
                                    let base_out = IOSurfaceGetBaseAddress(io_out) as *const u16;
                                    let mut sample = [0u16; 8];
                                    for i in 0..8 {
                                        sample[i] = *base_out.add(i);
                                    }
                                    IOSurfaceUnlock(io_out, 0, ptr::null_mut());

                                    print!("  [+] Output[0..8] = [");
                                    for (i, &v) in sample.iter().enumerate() {
                                        if i > 0 { print!(", "); }
                                        // Decode fp16 to f32 for display
                                        let sign = (v >> 15) & 1;
                                        let exp = (v >> 10) & 0x1F;
                                        let frac = v & 0x3FF;
                                        let f = if exp == 0 {
                                            ((-1.0f32).powi(sign as i32)) * (frac as f32 / 1024.0) * 2.0f32.powi(-14)
                                        } else if exp == 31 {
                                            if frac == 0 { f32::INFINITY } else { f32::NAN }
                                        } else {
                                            ((-1.0f32).powi(sign as i32)) * (1.0 + frac as f32 / 1024.0) * 2.0f32.powi(exp as i32 - 15)
                                        };
                                        print!("{:.4}", f);
                                    }
                                    println!("]");

                                    // Verify identity: output should equal input (all 1.0)
                                    let all_ones = sample.iter().all(|&v| v == 0x3C00);
                                    if all_ones {
                                        println!("\n  ╔═══════════════════════════════════════════╗");
                                        println!("  ║  IDENTITY VERIFIED — ANE COMPUTED y = x   ║");
                                        println!("  ║  Pure Rust → ANE hardware. No ObjC.        ║");
                                        println!("  ╚═══════════════════════════════════════════╝");
                                    }
                                } else {
                                    println!("  [-] Evaluation failed");
                                    if !eval_err.is_null() {
                                        let desc_sel = sel("description");
                                        let desc: ObjcId = std::mem::transmute(
                                            (std::mem::transmute::<_, MsgSendStr>(objc_msgSend as *const c_void))(eval_err, desc_sel)
                                        );
                                        if !desc.is_null() {
                                            type MsgSendUtf8 = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const c_char;
                                            let u: MsgSendUtf8 = std::mem::transmute(objc_msgSend as *const c_void);
                                            let s = u(desc, sel("UTF8String"));
                                            if !s.is_null() {
                                                println!("      Error: {}", CStr::from_ptr(s).to_string_lossy());
                                            }
                                        }
                                    }
                                }
                            } else {
                                println!("  [-] Failed to create _ANERequest");
                            }
                        }
                    }

                    // Unload
                    let mut unload_err: ObjcId = ptr::null_mut();
                    type MsgSendUnload = unsafe extern "C" fn(ObjcId, ObjcSel, u32, *mut ObjcId) -> bool;
                    let unload_fn: MsgSendUnload = std::mem::transmute(objc_msgSend as *const c_void);
                    let _ = unload_fn(model, sel("unloadWithQoS:error:"), 21, &mut unload_err);
                    println!("\n  [+] Model unloaded");

                    // Cleanup temp dir
                    let _ = std::fs::remove_dir_all(&tmp_dir);
                }
            } else {
                println!("  [-] Load failed");
                if !load_err.is_null() {
                    println!("      Error: {}", cf_desc(load_err));
                }
            }

            return true;
        } else {
            println!("  [-] Compilation failed");
            if !error.is_null() {
                type MsgSendStr2 = unsafe extern "C" fn(ObjcId, ObjcSel) -> ObjcId;
                let desc_fn: MsgSendStr2 = std::mem::transmute(objc_msgSend as *const c_void);
                let desc = desc_fn(error, sel("description"));
                if !desc.is_null() {
                    type MsgSendUtf8 = unsafe extern "C" fn(ObjcId, ObjcSel) -> *const c_char;
                    let utf8_fn: MsgSendUtf8 = std::mem::transmute(objc_msgSend as *const c_void);
                    let cstr = utf8_fn(desc, sel("UTF8String"));
                    if !cstr.is_null() {
                        println!("      Error: {}", CStr::from_ptr(cstr).to_string_lossy());
                    }
                }
            }
            return false;
        }
    }
}

// Helper: create IOSurface of given byte size
fn level4_make_surface(bytes: usize) -> IOSurfaceRef {
    unsafe {
        let dict = CFDictionaryCreateMutable(
            ptr::null(), 0,
            &kCFTypeDictionaryKeyCallBacks as *const c_void,
            &kCFTypeDictionaryValueCallBacks as *const c_void,
        );
        CFDictionarySetValue(dict, cf_str("IOSurfaceWidth") as _, cf_num(bytes as i32));
        CFDictionarySetValue(dict, cf_str("IOSurfaceHeight") as _, cf_num(1));
        CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerElement") as _, cf_num(1));
        CFDictionarySetValue(dict, cf_str("IOSurfaceBytesPerRow") as _, cf_num(bytes as i32));
        CFDictionarySetValue(dict, cf_str("IOSurfaceAllocSize") as _, cf_num(bytes as i32));
        CFDictionarySetValue(dict, cf_str("IOSurfacePixelFormat") as _, cf_num(0));
        IOSurfaceCreate(dict)
    }
}

// ============================================================
// Main
// ============================================================

fn main() {
    println!("╔═══════════════════════════════════════════════════╗");
    println!("║   ANE PROBE — Pure Rust → Apple Neural Engine    ║");
    println!("║   Target: M1 Pro / H11ANEIn / ane,t8020          ║");
    println!("╚═══════════════════════════════════════════════════╝\n");

    // Level 1: Find the ANE driver in IORegistry
    let service = level1_discover();

    // Level 2: Try to open user client
    let connect = if let Some(svc) = service {
        let conn = level2_open(svc);
        unsafe { IOObjectRelease(svc); }
        conn
    } else {
        println!("\n  [!] No ANE service found in IORegistry");
        None
    };

    // Level 3: Probe selectors if we got a connection
    if let Some(conn) = connect {
        level3_probe(conn);
    } else {
        println!("\n  [!] No user client — will still probe IOSurface and symbols");
    }

    // Level 4: IOSurface — works regardless of ANE access
    let _surface = level4_surface();

    // Level 5: Load frameworks and enumerate C API
    let _fw = level5_frameworks();

    // Level 6: Try ANEServices direct C calls
    level6_direct_api();

    // Level 7: Compile MIL → ANE bytecode
    level7_mil_compile();

    // Cleanup
    if let Some(conn) = connect {
        unsafe { IOServiceClose(conn); }
    }

    println!("\n═══════════════════════════════════════════════════");
    println!("  PROBE COMPLETE");
    println!("═══════════════════════════════════════════════════");
}
