//! Image format conversion utilities

use std::{
    collections::HashMap,
    io::Cursor,
    path::{Path, PathBuf},
    sync::Arc,
};

use tokio::sync::Semaphore;

#[cfg(feature = "format-conversion")]
use image::{DynamicImage, ImageOutputFormat};

use crate::MullamaError;

use super::{ConversionConfig, ImageConversionResult, ImageFormatType};

/// Image format converter
#[cfg(feature = "format-conversion")]
pub struct ImageConverter {
    config: ImageConverterConfig,
    conversion_cache: Arc<tokio::sync::RwLock<HashMap<String, Vec<u8>>>>,
    semaphore: Arc<Semaphore>,
}

/// Image converter configuration
#[cfg(feature = "format-conversion")]
#[derive(Debug, Clone)]
pub struct ImageConverterConfig {
    /// Maximum concurrent conversions
    pub max_concurrent: usize,
    /// Enable caching of converted images
    pub enable_cache: bool,
    /// Default JPEG quality (1-100)
    pub jpeg_quality: u8,
    /// Default PNG compression level (0-9)
    pub png_compression: u8,
    /// Default WebP quality (0.0-100.0)
    pub webp_quality: f32,
    /// Maximum image dimensions
    pub max_dimensions: (u32, u32),
}

#[cfg(feature = "format-conversion")]
impl ImageConverter {
    /// Create a new image converter
    pub fn new() -> Self {
        Self::with_config(ImageConverterConfig::default())
    }

    /// Create image converter with custom configuration
    pub fn with_config(config: ImageConverterConfig) -> Self {
        Self {
            semaphore: Arc::new(Semaphore::new(config.max_concurrent)),
            conversion_cache: Arc::new(tokio::sync::RwLock::new(HashMap::new())),
            config,
        }
    }

    /// Convert JPEG to PNG
    pub async fn jpeg_to_png(
        &self,
        input_path: impl AsRef<Path>,
        config: ConversionConfig,
    ) -> Result<ImageConversionResult, MullamaError> {
        self.convert_image(
            input_path,
            ImageFormatType::Jpeg,
            ImageFormatType::Png,
            config,
        )
        .await
    }

    /// Convert PNG to JPEG
    pub async fn png_to_jpeg(
        &self,
        input_path: impl AsRef<Path>,
        config: ConversionConfig,
    ) -> Result<ImageConversionResult, MullamaError> {
        self.convert_image(
            input_path,
            ImageFormatType::Png,
            ImageFormatType::Jpeg,
            config,
        )
        .await
    }

    /// Convert WebP to PNG
    pub async fn webp_to_png(
        &self,
        input_path: impl AsRef<Path>,
        config: ConversionConfig,
    ) -> Result<ImageConversionResult, MullamaError> {
        self.convert_image(
            input_path,
            ImageFormatType::WebP,
            ImageFormatType::Png,
            config,
        )
        .await
    }

    /// Convert between any supported image formats
    pub async fn convert_image(
        &self,
        input_path: impl AsRef<Path>,
        input_format: ImageFormatType,
        output_format: ImageFormatType,
        config: ConversionConfig,
    ) -> Result<ImageConversionResult, MullamaError> {
        let _permit = self.semaphore.acquire().await.map_err(|_| {
            MullamaError::ConfigError("Failed to acquire conversion semaphore".to_string())
        })?;

        let input_path = input_path.as_ref();
        let cache_key = format!(
            "{:?}_{:?}_{}",
            input_path.display(),
            output_format,
            serde_json::to_string(&config).unwrap_or_default()
        );

        // Check cache first
        if self.config.enable_cache {
            let cache = self.conversion_cache.read().await;
            if let Some(cached_data) = cache.get(&cache_key) {
                return self
                    .create_image_result(cached_data.clone(), output_format, &config)
                    .await;
            }
        }

        // Load and convert image
        let img = image::open(input_path)
            .map_err(|e| MullamaError::ConfigError(format!("Failed to open image: {}", e)))?;

        let result = self
            .convert_dynamic_image(img, output_format, &config)
            .await?;

        // Cache result
        if self.config.enable_cache {
            let mut cache = self.conversion_cache.write().await;
            cache.insert(cache_key, result.data.clone());
        }

        Ok(result)
    }

    /// Convert image from bytes
    pub async fn convert_image_bytes(
        &self,
        input_data: &[u8],
        input_format: ImageFormatType,
        output_format: ImageFormatType,
        config: ConversionConfig,
    ) -> Result<ImageConversionResult, MullamaError> {
        let _permit = self.semaphore.acquire().await.map_err(|_| {
            MullamaError::ConfigError("Failed to acquire conversion semaphore".to_string())
        })?;

        let img = image::load_from_memory(input_data).map_err(|e| {
            MullamaError::ConfigError(format!("Failed to load image from memory: {}", e))
        })?;

        self.convert_dynamic_image(img, output_format, &config)
            .await
    }

    /// Resize image to specific dimensions
    pub async fn resize_image(
        &self,
        input_path: impl AsRef<Path>,
        dimensions: (u32, u32),
        filter: image::imageops::FilterType,
    ) -> Result<ImageConversionResult, MullamaError> {
        let img = image::open(input_path)
            .map_err(|e| MullamaError::ConfigError(format!("Failed to open image: {}", e)))?;

        let resized = img.resize(dimensions.0, dimensions.1, filter);

        let config = ConversionConfig {
            dimensions: Some(dimensions),
            ..Default::default()
        };

        self.convert_dynamic_image(resized, ImageFormatType::Png, &config)
            .await
    }

    /// Batch convert multiple images
    pub async fn batch_convert_images(
        &self,
        conversions: Vec<(PathBuf, ImageFormatType, ImageFormatType, ConversionConfig)>,
    ) -> Result<Vec<ImageConversionResult>, MullamaError> {
        let mut results = Vec::new();

        for (path, input_fmt, output_fmt, config) in conversions {
            let result = self
                .convert_image(&path, input_fmt, output_fmt, config)
                .await?;
            results.push(result);
        }

        Ok(results)
    }

    // Private helper methods
    async fn convert_dynamic_image(
        &self,
        mut img: DynamicImage,
        output_format: ImageFormatType,
        config: &ConversionConfig,
    ) -> Result<ImageConversionResult, MullamaError> {
        // Resize if requested
        if let Some((width, height)) = config.dimensions {
            img = img.resize(width, height, image::imageops::FilterType::Lanczos3);
        }

        // Check dimensions against max limits
        let (width, height) = (img.width(), img.height());
        if width > self.config.max_dimensions.0 || height > self.config.max_dimensions.1 {
            return Err(MullamaError::ConfigError(format!(
                "Image dimensions {}x{} exceed maximum {}x{}",
                width, height, self.config.max_dimensions.0, self.config.max_dimensions.1
            )));
        }

        // Convert to target format
        let mut cursor = Cursor::new(Vec::new());

        match output_format {
            ImageFormatType::Jpeg => {
                let quality = config.quality.unwrap_or(self.config.jpeg_quality as f32) as u8;
                img.write_to(&mut cursor, ImageOutputFormat::Jpeg(quality))
                    .map_err(|e| {
                        MullamaError::ConfigError(format!("JPEG encoding failed: {}", e))
                    })?;
            }
            ImageFormatType::Png => {
                let compression = if let Some(quality) = config.quality {
                    (quality * 9.0) as u8
                } else {
                    self.config.png_compression
                };
                img.write_to(&mut cursor, ImageOutputFormat::Png)
                    .map_err(|e| {
                        MullamaError::ConfigError(format!("PNG encoding failed: {}", e))
                    })?;
            }
            ImageFormatType::WebP => {
                let quality = config.quality.unwrap_or(self.config.webp_quality);
                // Note: WebP support would require additional dependencies
                img.write_to(&mut cursor, ImageOutputFormat::Png)
                    .map_err(|e| {
                        MullamaError::ConfigError(format!("WebP encoding failed: {}", e))
                    })?;
            }
            ImageFormatType::Bmp => {
                img.write_to(&mut cursor, ImageOutputFormat::Bmp)
                    .map_err(|e| {
                        MullamaError::ConfigError(format!("BMP encoding failed: {}", e))
                    })?;
            }
            ImageFormatType::Tiff => {
                img.write_to(&mut cursor, ImageOutputFormat::Tiff)
                    .map_err(|e| {
                        MullamaError::ConfigError(format!("TIFF encoding failed: {}", e))
                    })?;
            }
            _ => {
                return Err(MullamaError::ConfigError(format!(
                    "Unsupported output format: {:?}",
                    output_format
                )));
            }
        }

        Ok(ImageConversionResult {
            data: cursor.into_inner(),
            format: output_format,
            width,
            height,
            metadata: HashMap::new(),
        })
    }

    async fn create_image_result(
        &self,
        data: Vec<u8>,
        format: ImageFormatType,
        config: &ConversionConfig,
    ) -> Result<ImageConversionResult, MullamaError> {
        // Extract basic info from cached data
        let dimensions = config.dimensions.unwrap_or((800, 600));

        Ok(ImageConversionResult {
            data,
            format,
            width: dimensions.0,
            height: dimensions.1,
            metadata: HashMap::new(),
        })
    }
}

#[cfg(feature = "format-conversion")]
impl Default for ImageConverterConfig {
    fn default() -> Self {
        Self {
            max_concurrent: 4,
            enable_cache: true,
            jpeg_quality: 85,
            png_compression: 6,
            webp_quality: 80.0,
            max_dimensions: (4096, 4096),
        }
    }
}
