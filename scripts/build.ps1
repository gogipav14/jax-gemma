<#
  Build the CnCNet spawner / bridge DLL (Debug|Win32) with VS2022 BuildTools.
  Usage:  pwsh -File scripts\build.ps1 [-Config Debug] [-Clean]
  Output: bridge\yrpp-spawner\Debug\CnCNet-Spawner.dll
#>
param(
  [string]$Config = "Debug",   # Debug | Release (Win32 platform is implied; YR is 32-bit)
  [switch]$Clean
)

$ErrorActionPreference = "Stop"
$msbuild = "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Current\Bin\MSBuild.exe"
$proj    = Join-Path $PSScriptRoot "..\bridge\yrpp-spawner\Spawner.vcxproj"
$proj    = (Resolve-Path $proj).Path

if (-not (Test-Path $msbuild)) { throw "MSBuild not found at $msbuild" }

# Preflight: ATL must be installed (the common Phase 0 blocker).
$atl = Get-ChildItem 'C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC' `
        -Recurse -Filter 'atlbase.h' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $atl) {
  Write-Warning "ATL (atlbase.h) is NOT installed. Add it via the VS Installer:"
  Write-Warning "  Visual Studio Installer -> Build Tools 2022 -> Modify -> Individual components"
  Write-Warning "  -> 'C++ ATL for latest v143 build tools (x86 & x64)' -> Modify"
  throw "Missing ATL component."
}

$targets = if ($Clean) { "Rebuild" } else { "Build" }
& $msbuild $proj /t:$targets /p:Configuration=$Config /p:Platform=Win32 /m /v:minimal /nologo /clp:Summary
if ($LASTEXITCODE -ne 0) { throw "Build failed (exit $LASTEXITCODE)." }

$dll = Join-Path (Split-Path $proj) "$Config\CnCNet-Spawner.dll"
if (Test-Path $dll) { Write-Host "BUILD OK -> $dll" -ForegroundColor Green }
else { Write-Warning "Build reported success but DLL not found at $dll" }
