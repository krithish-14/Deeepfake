<#
PowerShell helper to push repository to remote with safe pull/rebase.
Usage examples:
  # Normal push (will pull --rebase first if remote exists)
  .\git_push.ps1 -RemoteUrl 'https://github.com/krithish-14/Deeepfake.git' -Branch main

  # Force push (use only if you understand remote history will be overwritten)
  .\git_push.ps1 -RemoteUrl 'https://github.com/krithish-14/Deeepfake.git' -Branch main -Force
#>
param(
    [Parameter(Mandatory=$true)] [string]$RemoteUrl,
    [string]$Branch = 'main',
    [switch]$Force,
    [string]$CommitMessage = "Auto-update: push from helper script"
)

function Run-Git([string]$args) {
    Write-Host "git $args"
    & git $args
    if ($LASTEXITCODE -ne 0) {
        throw "git command failed: git $args"
    }
}

# ensure we're in repo root
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location (Resolve-Path "$scriptDir\..")

# init if necessary
try {
    git rev-parse --is-inside-work-tree > $null 2>&1
} catch {
    Write-Host "Repository not initialized. Initializing git repository..."
    Run-Git 'init'
    Run-Git "branch -M $Branch"
}

# ensure remote is set
$existing = git remote get-url origin 2>$null
if ($?) {
    Write-Host "Existing origin remote: $existing"
    if ($existing -ne $RemoteUrl) {
        Write-Host "Updating origin to $RemoteUrl"
        Run-Git 'remote remove origin'
        Run-Git "remote add origin $RemoteUrl"
    }
} else {
    Write-Host "Adding origin remote: $RemoteUrl"
    Run-Git "remote add origin $RemoteUrl"
}

# Add and commit any changes
Run-Git 'add -A'
# Only commit if there are staged changes
$diffIndex = git diff --cached --name-only
if ($diffIndex.Trim().Length -gt 0) {
    try {
        Run-Git "commit -m \"$CommitMessage\""
    } catch {
        Write-Host "Commit failed (possibly empty commit); continuing."
    }
} else {
    Write-Host "No changes to commit."
}

# Fetch latest
Run-Git 'fetch origin'

if (-not $Force) {
    Write-Host "Attempting safe pull --rebase from origin/$Branch"
    try {
        Run-Git "pull --rebase origin $Branch"
    } catch {
        Write-Host "Pull --rebase failed. Resolve conflicts manually or rerun with -Force to overwrite remote."
        throw $_
    }
}

# Push
if ($Force) {
    Write-Host "Force pushing to origin/$Branch"
    Run-Git "push --force origin $Branch"
} else {
    Write-Host "Pushing to origin/$Branch"
    Run-Git "push origin $Branch"
}

Write-Host "Push complete."