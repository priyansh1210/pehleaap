# Point Claude Code's auto memory at this repo's memory/ folder, so both machines share one memory through git.
# Run once per machine, from anywhere:
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_claude_memory.ps1
#
# Writes autoMemoryDirectory into .claude\settings.local.json (gitignored, per machine). It can't live in the
# committed .claude\settings.json: Claude Code ignores the setting there, and the repo path differs per machine.

$Repo = (Resolve-Path "$PSScriptRoot\..").Path
$Memory = (Join-Path $Repo "memory") -replace '\\', '/'
$SettingsDir = Join-Path $Repo ".claude"
$Settings = Join-Path $SettingsDir "settings.local.json"

New-Item -ItemType Directory -Force $SettingsDir | Out-Null
$Config = if (Test-Path $Settings) { Get-Content $Settings -Raw | ConvertFrom-Json } else { New-Object psobject }
$Config | Add-Member -NotePropertyName autoMemoryDirectory -NotePropertyValue $Memory -Force

# UTF-8 without a BOM, so every JSON reader accepts it.
[IO.File]::WriteAllText($Settings, ($Config | ConvertTo-Json -Depth 20), (New-Object Text.UTF8Encoding $false))
Write-Host "Claude Code auto memory on this machine -> $Memory"
Write-Host "Restart Claude Code in this folder to pick it up."
