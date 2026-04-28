param(
    [switch]$SkipBuild
)

$ErrorActionPreference = "Stop"

if (-not $SkipBuild) {
    Write-Host "Building DMT CLI (ReleaseFast) for Windows x86_64..."
    zig build -Drelease=true -Dtarget=x86_64-windows
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Zig build failed."
        exit 1
    }
}

Write-Host "Compiling MSI using WiX Toolset..."
.\wix\candle.exe -arch x64 -ext WixUIExtension installer.wxs
if ($LASTEXITCODE -ne 0) {
    Write-Error "WiX candle failed."
    exit 1
}

Write-Host "Linking MSI..."
.\wix\light.exe -ext WixUIExtension -out dmt-cli-installer.msi installer.wixobj
if ($LASTEXITCODE -ne 0) {
    Write-Error "WiX light failed."
    exit 1
}

Write-Host "Success! Installer created at dmt-cli-installer.msi"
