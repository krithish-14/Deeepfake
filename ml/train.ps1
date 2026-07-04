param(
    [string]$DataDir = ".\datasets\ffpp",
    [int]$Epochs = 10,
    [int]$BatchSize = 16,
    [double]$LR = 1e-4
)

$scriptFolder = Split-Path -Parent $MyInvocation.MyCommand.Definition
$repoRoot = Split-Path -Parent $scriptFolder
Set-Location -Path $scriptFolder

$activatePath = Join-Path $repoRoot ".venv\Scripts\Activate.ps1"
if (Test-Path $activatePath) {
    Write-Host "Activating venv: $activatePath"
    & $activatePath
} else {
    Write-Host "No virtualenv activation script found at $activatePath. Using system Python."
}

python .\train_real.py --data-dir $DataDir --epochs $Epochs --batch-size $BatchSize --lr $LR
