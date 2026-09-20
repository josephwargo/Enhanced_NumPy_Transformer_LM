#Requires -Version 5.1
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string]$Branch
)

$ErrorActionPreference = 'Stop'

# git is a native exe, so it won't throw on failure - check the exit code ourselves
function Invoke-Git {
    param([Parameter(ValueFromRemainingArguments = $true)]$GitArgs)
    & git @GitArgs
    if ($LASTEXITCODE -ne 0) {
        throw "git $($GitArgs -join ' ') failed (exit $LASTEXITCODE)"
    }
}

# move to the repo root
$root = & git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0) { throw "not inside a git repository" }
Set-Location $root

if ([string]::IsNullOrWhiteSpace($Branch)) {
    # no argument: use whatever branch we're already on
    $Branch = & git symbolic-ref --quiet --short HEAD
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($Branch)) {
        throw "detached HEAD - pass a branch name explicitly"
    }
    Write-Host "no branch given, using current branch '$Branch'"
}
else {
    # argument given: must already exist locally
    & git show-ref --verify --quiet "refs/heads/$Branch"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "branch '$Branch' does not exist locally"
        Write-Host "existing branches:"
        & git branch
        exit 1
    }
    Invoke-Git switch $Branch
}

Invoke-Git add -A

# exit code 0 means no staged changes
& git diff --cached --quiet
if ($LASTEXITCODE -eq 0) {
    Write-Host "nothing to commit"
}
else {
    Invoke-Git commit -m "debugging"
}

Invoke-Git push -u origin $Branch
& git log -1 --oneline