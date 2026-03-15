<?php

declare(strict_types=1);

namespace Mullama;

use FFI;
use RuntimeException;

/**
 * Main Mullama class that handles FFI library loading and backend management.
 */
final class Mullama
{
    private static ?FFI $ffi = null;
    private static bool $initialized = false;
    private static ?string $libraryPath = null;

    /**
     * Get the FFI instance, initializing if necessary.
     */
    public static function ffi(): FFI
    {
        if (self::$ffi === null) {
            self::initialize();
        }
        return self::$ffi;
    }

    /**
     * Set a custom library path before initialization.
     */
    public static function setLibraryPath(string $path): void
    {
        if (self::$ffi !== null) {
            throw new RuntimeException('Cannot set library path after initialization');
        }
        self::$libraryPath = $path;
    }

    /**
     * Initialize the Mullama backend.
     */
    public static function initialize(): void
    {
        if (self::$ffi !== null) {
            return;
        }

        $headerPath = self::findHeaderPath();
        $libPath = self::$libraryPath ?? self::findLibraryPath();

        if (!file_exists($headerPath)) {
            throw new RuntimeException("Header file not found: {$headerPath}");
        }

        if (!file_exists($libPath)) {
            throw new RuntimeException("Library not found: {$libPath}");
        }

        // Read and preprocess the header
        $header = self::preprocessHeader($headerPath);

        self::$ffi = FFI::cdef($header, $libPath);
        self::$ffi->mullama_backend_init();
        self::$initialized = true;
    }

    /**
     * Free the Mullama backend resources.
     */
    public static function shutdown(): void
    {
        if (self::$initialized && self::$ffi !== null) {
            self::$ffi->mullama_backend_free();
            self::$initialized = false;
        }
    }

    /**
     * Check if GPU offloading is supported.
     */
    public static function supportsGpuOffload(): bool
    {
        return (bool) self::ffi()->mullama_supports_gpu_offload();
    }

    /**
     * Get system information.
     */
    public static function systemInfo(): string
    {
        $ffi = self::ffi();
        $buf = FFI::new('char[4096]');
        $n = $ffi->mullama_system_info(FFI::addr($buf[0]), 4096);
        if ($n < 0) {
            return '';
        }
        return FFI::string($buf);
    }

    /**
     * Get maximum number of devices.
     */
    public static function maxDevices(): int
    {
        return (int) self::ffi()->mullama_max_devices();
    }

    /**
     * Get the last error message.
     */
    public static function getLastError(): string
    {
        $ptr = self::ffi()->mullama_get_last_error();
        if ($ptr === null) {
            return 'Unknown error';
        }
        return FFI::string($ptr);
    }

    /**
     * Get library version.
     */
    public static function version(): string
    {
        return '0.1.0';
    }

    /**
     * Find the header file path.
     */
    private static function findHeaderPath(): string
    {
        $candidates = [
            __DIR__ . '/../../ffi/include/mullama.h',
            __DIR__ . '/../../../bindings/ffi/include/mullama.h',
            '/usr/local/include/mullama.h',
            '/usr/include/mullama.h',
        ];

        foreach ($candidates as $path) {
            $realPath = realpath($path);
            if ($realPath !== false && file_exists($realPath)) {
                return $realPath;
            }
        }

        throw new RuntimeException('Could not find mullama.h header file');
    }

    /**
     * Find the library path.
     */
    private static function findLibraryPath(): string
    {
        $libName = PHP_OS_FAMILY === 'Darwin' ? 'libmullama_ffi.dylib' : 'libmullama_ffi.so';

        $candidates = [
            __DIR__ . '/../../../target/release/' . $libName,
            __DIR__ . '/../../lib/' . $libName,
            '/usr/local/lib/' . $libName,
            '/usr/lib/' . $libName,
        ];

        foreach ($candidates as $path) {
            $realPath = realpath($path);
            if ($realPath !== false && file_exists($realPath)) {
                return $realPath;
            }
        }

        throw new RuntimeException("Could not find {$libName} library file");
    }

    /**
     * Preprocess the header file for FFI.
     */
    private static function preprocessHeader(string $path): string
    {
        $content = file_get_contents($path);

        // Remove #ifndef/#define/#endif guards
        $content = preg_replace('/#ifndef\s+\w+\s*\n/', '', $content);
        $content = preg_replace('/#define\s+\w+\s*\n/', '', $content);
        $content = preg_replace('/#endif.*\n/', '', $content);

        // Remove #includes
        $content = preg_replace('/#include\s*[<"][^>"]+[>"]\s*\n/', '', $content);

        // Remove #ifdef __cplusplus blocks
        $content = preg_replace('/#ifdef\s+__cplusplus.*?#endif.*?\n/s', '', $content);

        // Remove single-line comments with #
        $content = preg_replace('/^#.*$/m', '', $content);

        // Remove /* autogenerated */ warnings
        $content = preg_replace('/\/\*.*?\*\//s', '', $content);

        // Remove typedef struct { volatile int value; } MullamaAtomicBool; - already defined
        $content = str_replace('typedef struct { volatile int value; } MullamaAtomicBool;', '', $content);

        // Add necessary typedefs
        $preamble = "
typedef struct { volatile int value; } MullamaAtomicBool;
";

        return $preamble . $content;
    }
}
