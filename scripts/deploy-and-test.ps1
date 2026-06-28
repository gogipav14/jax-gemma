<#
  Deploy the freshly built bridge DLL into the CnCNet YR install and optionally launch the client
  to test injection. Backs up the original spawner DLL first. Reversible with -Restore.

  Usage:
    pwsh -File scripts\deploy-and-test.ps1            # back up + deploy dev DLL
    pwsh -File scripts\deploy-and-test.ps1 -Launch    # ...and start the CnCNet client
    pwsh -File scripts\deploy-and-test.ps1 -Restore   # revert to the original hardened DLL
#>
param([switch]$Launch, [switch]$Restore)
$ErrorActionPreference = "Stop"

$game   = "C:\Program Files (x86)\Steam\steamapps\common\Command & Conquer Red Alert II"
$built  = Join-Path $PSScriptRoot "..\bridge\yrpp-spawner\Debug\CnCNet-Spawner.dll"
$target = Join-Path $game "CnCNet-Spawner.dll"
$backup = Join-Path $game "CnCNet-Spawner.dll.orig"

if ($Restore) {
  if (Test-Path $backup) { Copy-Item $backup $target -Force; Write-Host "Restored original spawner DLL." -ForegroundColor Green }
  else { Write-Warning "No backup ($backup) found - nothing to restore." }
  return
}

if (-not (Test-Path $built)) { throw "Built DLL not found: $built  (run scripts\build.ps1 first)" }
if (-not (Test-Path $backup)) { Copy-Item $target $backup -Force; Write-Host "Backed up original -> $backup" -ForegroundColor Yellow }
Copy-Item $built $target -Force
Write-Host "Deployed dev DLL -> $target" -ForegroundColor Green

if ($Launch) {
  # Correct CnCNet-matching invocation (from Client\client.log / QuickMatch.ini).
  # Game args MUST be inside --args="..."; bare flags get eaten by SyringeEx.
  # -LOG makes Ares/Phobos/Spawner write debug\debug.log (needed for our Bridge dumps).
  $syringe = Join-Path $game "Syringe.exe"
  $argLine = '-i=Ares.dll -i=CnCNet-Spawner.dll -i=Phobos.dll gamemd-spawn.exe --args="-SPAWN -LOG -CD -Include -Inheritance"'
  Write-Host "Launching skirmish via Syringe (uses the existing spawn.ini)..."
  Start-Process -FilePath $syringe -ArgumentList $argLine -WorkingDirectory $game
}
