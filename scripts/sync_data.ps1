# Mirror a data folder between the Dev box and the GPU box.
#
#   .\scripts\sync_data.ps1 raw              # push data\raw (recordings) to the GPU box, mirroring deletions
#   .\scripts\sync_data.ps1 labels -Pull     # pull data\labels (ASR + auto labels) back, never deleting locally
#
# $env:PEHLEAAP_GPU_SHARE is the GPU box's data folder: an SMB share (\\GPUBOX\pehleaap-data)
# or a portable SSD path (E:\pehleaap-data). After pushing, verify on the GPU box with
#   python scripts\env_check.py --verify-data
param(
    [Parameter(Mandatory, Position = 0)][string]$Subdir,
    [switch]$Pull
)

$Remote = $env:PEHLEAAP_GPU_SHARE
if (-not $Remote) { throw "Set PEHLEAAP_GPU_SHARE to the GPU box's data folder, e.g. \\GPUBOX\pehleaap-data" }
$Local = Join-Path (Resolve-Path "$PSScriptRoot\..") "data\$Subdir"

if ($Pull) {
    robocopy (Join-Path $Remote $Subdir) $Local /E /Z /R:3 /W:5 /NP /NDL
} else {
    robocopy $Local (Join-Path $Remote $Subdir) /MIR /Z /R:3 /W:5 /NP /NDL
}
# robocopy exit codes 0-7 mean success (with or without copies); 8+ means failures.
if ($LASTEXITCODE -ge 8) { throw "robocopy failed with exit code $LASTEXITCODE" }
exit 0
