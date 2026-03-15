# Mullama installer script for Windows
# Usage: iwr https://mullama.dev/install.ps1 -useb | iex

$ErrorActionPreference = "Stop"

# Configuration
$Repo = "neul-labs/mullama"
$BinaryName = "mullama"
$InstallDir = if ($env:MULLAMA_INSTALL_DIR) { $env:MULLAMA_INSTALL_DIR } else { "$env:LOCALAPPDATA\Programs\Mullama" }

# Colors
function Write-Info { Write-Host "[INFO] $args" -ForegroundColor Blue }
function Write-Success { Write-Host "[OK] $args" -ForegroundColor Green }
function Write-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Write-Err { Write-Host "[ERROR] $args" -ForegroundColor Red; exit 1 }

# Detect architecture
function Get-Platform {
    $arch = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    switch ($arch) {
        "X64" { return "win32-x64" }
        "Arm64" { return "win32-arm64" }
        default { Write-Err "Unsupported architecture: $arch" }
    }
}

# Get latest version from GitHub
function Get-LatestVersion {
    Write-Info "Fetching latest version..."
    try {
        $release = Invoke-RestMethod -Uri "https://api.github.com/repos/$Repo/releases/latest" -Headers @{ "User-Agent" = "Mullama-Installer" }
        $version = $release.tag_name -replace "^v", ""
        Write-Info "Latest version: v$version"
        return $version
    } catch {
        Write-Err "Failed to get latest version: $_"
    }
}

# Download and extract
function Install-Mullama {
    param($Version, $Platform)

    $filename = "$BinaryName-$Platform.zip"
    $downloadUrl = "https://github.com/$Repo/releases/download/v$Version/$filename"
    $tempDir = Join-Path $env:TEMP "mullama-install-$(Get-Random)"

    try {
        # Create temp directory
        New-Item -ItemType Directory -Path $tempDir -Force | Out-Null

        # Download
        Write-Info "Downloading $filename..."
        $zipPath = Join-Path $tempDir $filename
        Invoke-WebRequest -Uri $downloadUrl -OutFile $zipPath -UseBasicParsing

        Write-Success "Downloaded successfully"

        # Extract
        Write-Info "Extracting..."
        Expand-Archive -Path $zipPath -DestinationPath $tempDir -Force

        # Create install directory
        if (-not (Test-Path $InstallDir)) {
            New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
        }

        # Install
        Write-Info "Installing to $InstallDir..."
        $exePath = Join-Path $tempDir "$BinaryName.exe"
        Copy-Item -Path $exePath -Destination $InstallDir -Force

        Write-Success "Installed $BinaryName to $InstallDir"
    } finally {
        # Cleanup
        if (Test-Path $tempDir) {
            Remove-Item -Path $tempDir -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

# Add to PATH
function Update-Path {
    $currentPath = [Environment]::GetEnvironmentVariable("Path", "User")

    if ($currentPath -notlike "*$InstallDir*") {
        Write-Info "Adding $InstallDir to PATH..."
        $newPath = "$currentPath;$InstallDir"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")

        # Also update current session
        $env:Path = "$env:Path;$InstallDir"

        Write-Success "Added to PATH"
        Write-Warn "You may need to restart your terminal for PATH changes to take effect"
    } else {
        Write-Info "$InstallDir is already in PATH"
    }
}

# Verify installation
function Test-Installation {
    $exePath = Join-Path $InstallDir "$BinaryName.exe"
    if (Test-Path $exePath) {
        try {
            $version = & $exePath --version 2>&1
            Write-Success "Installation verified: $version"
            return $true
        } catch {
            Write-Warn "Binary exists but couldn't get version"
            return $true
        }
    }
    return $false
}

# Print success message
function Show-SuccessMessage {
    Write-Host ""
    Write-Host "============================================" -ForegroundColor Green
    Write-Host "   Mullama installed successfully!" -ForegroundColor Green
    Write-Host "============================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Quick start:"
    Write-Host "  mullama run llama3.2:1b    # Run a model"
    Write-Host "  mullama serve              # Start the daemon"
    Write-Host "  mullama chat               # Interactive chat"
    Write-Host ""
    Write-Host "For more information, visit:"
    Write-Host "  https://github.com/$Repo"
    Write-Host ""
}

# Main
function Main {
    Write-Host ""
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host "   Mullama Installer" -ForegroundColor Cyan
    Write-Host "================================" -ForegroundColor Cyan
    Write-Host ""

    $platform = Get-Platform
    Write-Info "Detected platform: $platform"

    $version = Get-LatestVersion
    Install-Mullama -Version $version -Platform $platform
    Update-Path

    if (Test-Installation) {
        Show-SuccessMessage
    } else {
        Write-Err "Installation verification failed"
    }
}

Main
