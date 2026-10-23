# Launch a pehleaap module on the GPU box from the Dev box.
#
#   .\scripts\remote.ps1 train.train_eot --config configs\train_eot.yaml           # foreground
#   .\scripts\remote.ps1 -Detach train.train_eot --config configs\train_eot.yaml   # survives logout
#
# Needs OpenSSH Server on the GPU box and an ssh alias (default "gpu") in ~/.ssh/config.
# Override with $env:PEHLEAAP_GPU_HOST and $env:PEHLEAAP_GPU_REPO.
param(
    [switch]$Detach,
    [Parameter(Mandatory, Position = 0)][string]$Module,
    [Parameter(ValueFromRemainingArguments)][string[]]$Rest = @()
)

$GpuHost = if ($env:PEHLEAAP_GPU_HOST) { $env:PEHLEAAP_GPU_HOST } else { "gpu" }
$Repo = if ($env:PEHLEAAP_GPU_REPO) { $env:PEHLEAAP_GPU_REPO } else { "D:\pehleaap" }
$Job = "$Repo\scripts\run_job.cmd $Module $($Rest -join ' ')"

if (-not $Detach) {
    ssh $GpuHost $Job
    exit $LASTEXITCODE
}

# Processes started over Windows OpenSSH are killed when the session closes, so long jobs are
# handed to Task Scheduler on the GPU box instead.
$Task = "pehleaap_" + ($Module -replace '\W', '_')
ssh $GpuHost "schtasks /Create /F /TN $Task /SC ONCE /ST 23:59 /TR `"$Job`" && schtasks /Run /TN $Task"
Write-Host "Started task $Task on $GpuHost. Log: $Repo\runs\last_$Module.log"
