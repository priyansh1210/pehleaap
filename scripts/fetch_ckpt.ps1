# Copy a finished run's small artifacts (ONNX, best checkpoint, metrics) from the GPU box.
#
#   .\scripts\fetch_ckpt.ps1 eot_v0_20261112_2130
param([Parameter(Mandatory, Position = 0)][string]$RunId)

$GpuHost = if ($env:PEHLEAAP_GPU_HOST) { $env:PEHLEAAP_GPU_HOST } else { "gpu" }
$Repo = if ($env:PEHLEAAP_GPU_REPO) { $env:PEHLEAAP_GPU_REPO } else { "C:\Users\FITT\pehleaap" }
$Dest = Join-Path (Resolve-Path "$PSScriptRoot\..") "runs\$RunId"
New-Item -ItemType Directory -Force $Dest | Out-Null

foreach ($f in "meta.json", "metrics.csv", "best.pt", "model.onnx", "model_int8.onnx") {
    scp "${GpuHost}:$($Repo -replace '\\', '/')/runs/$RunId/$f" $Dest 2>$null
    if ($LASTEXITCODE -eq 0) { Write-Host "fetched $f" }
}
