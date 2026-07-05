# Open Agent — Docker Quick Start demo script (Windows PowerShell)
#
# Flow:
#   1. Wait for the API to become ready (poll /api/ready until 200)
#   2. Upload sample-docs to the knowledge base (POST /api/upload)
#   3. Send a chat query "What is Acme Corp?" (POST /api/chat)
#   4. Print the response
#
# Usage:
#   .\init-and-query.ps1
#
# Override defaults via parameters or environment variables, e.g.:
#   .\init-and-query.ps1 -BaseUrl http://localhost:8000 -AuthToken demo-token-please-change

[CmdletBinding()]
param(
    [string]$BaseUrl     = $(if ($env:OPEN_AGENT_BASE_URL) { $env:OPEN_AGENT_BASE_URL } else { 'http://localhost:8000' }),
    [string]$AuthToken   = $(if ($env:OPEN_AGENT_API_AUTH_TOKEN) { $env:OPEN_AGENT_API_AUTH_TOKEN } else { 'demo-token-please-change' }),
    [string]$SampleDir   = $(if ($env:SAMPLE_DIR) { $env:SAMPLE_DIR } else { (Join-Path $PSScriptRoot 'sample-docs') }),
    [int]   $ReadyTimeout  = 120,
    [int]   $ReadyInterval = 3,
    [int]   $UploadTimeout = 60,
    [int]   $ChatTimeout   = 120
)

$ErrorActionPreference = 'Stop'

# Make sure System.Net.Http is loaded (needed for multipart upload on Windows PowerShell 5.1).
try { Add-Type -AssemblyName System.Net.Http } catch { }

function Write-Step($msg) { Write-Host "[$(Get-Date -Format 'HH:mm:ss')] $msg" -ForegroundColor White }
function Write-Ok($msg)   { Write-Host $msg -ForegroundColor Green }
function Write-Info($msg) { Write-Host $msg -ForegroundColor Cyan }
function Write-Warn($msg) { Write-Host $msg -ForegroundColor Yellow }
function Write-Err2($msg) { Write-Host "ERROR: $msg" -ForegroundColor Red }

# ---- Preflight checks ----
if (-not (Test-Path $SampleDir -PathType Container)) {
    Write-Err2 "Sample docs directory not found: $SampleDir"
    exit 1
}

$headers = @{ 'Authorization' = "Bearer $AuthToken" }

# ---- Step 1: Wait for the service to become ready ----
Write-Step "Waiting for Open Agent at $BaseUrl/api/ready (up to $ReadyTimeout s)..."
$deadline = (Get-Date).AddSeconds($ReadyTimeout)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $resp = Invoke-WebRequest -Uri "$BaseUrl/api/ready" -Method GET -TimeoutSec 5 -UseBasicParsing
        if ($resp.StatusCode -eq 200) { $ready = $true; break }
    } catch {
        # 503 (not ready) or connection refused -> keep waiting
        Write-Host -NoNewline "."
    }
    Start-Sleep -Seconds $ReadyInterval
}
Write-Host ""
if (-not $ready) {
    Write-Err2 "Service did not become ready within $ReadyTimeout s. Check 'docker compose logs'."
    exit 1
}
Write-Ok "Service is ready."

# ---- Step 2: Upload sample documents ----
Write-Info "Uploading sample documents from $SampleDir ..."
$uploaded = 0
$failed   = 0
Get-ChildItem -Path $SampleDir -File | ForEach-Object {
    $file = $_
    Write-Step "  - uploading $($file.Name) ..."
    try {
        # Build a multipart/form-data request with System.Net.Http.
        $form = New-Object System.Net.Http.MultipartFormDataContent
        $fileBytes  = [System.IO.File]::ReadAllBytes($file.FullName)
        $fileContent = New-Object System.Net.Http.ByteArrayContent(,$fileBytes)
        $fileContent.Headers.ContentType = New-Object System.Net.Http.MediaTypeHeaderValue('application/octet-stream')
        $form.Add($fileContent, 'file', $file.Name)
        $kbField = New-Object System.Net.Http.StringContent('default')
        $form.Add($kbField, 'kb_name')

        $client = New-Object System.Net.Http.HttpClient
        $client.Timeout = [TimeSpan]::FromSeconds($UploadTimeout)
        $client.DefaultRequestHeaders.Add('Authorization', "Bearer $AuthToken")
        $response = $client.PostAsync("$BaseUrl/api/upload", $form).Result
        $body = $response.Content.ReadAsStringAsync().Result
        $client.Dispose()
        if ($response.IsSuccessStatusCode) {
            Write-Ok "  uploaded $($file.Name) (HTTP $([int]$response.StatusCode)): $body"
            $uploaded++
        } else {
            Write-Warn "  failed to upload $($file.Name) (HTTP $([int]$response.StatusCode)): $body"
            $failed++
        }
    } catch {
        Write-Warn "  failed to upload $($file.Name): $($_.Exception.Message)"
        $failed++
    }
}

if ($uploaded -eq 0) {
    Write-Err2 "No documents were uploaded. Cannot proceed with RAG query."
    exit 1
}
Write-Warn "Uploaded $uploaded file(s); failed: $failed."

# ---- Step 3: Send a chat query ----
$question  = 'What is Acme Corp?'
$sessionId = "demo-$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())"
Write-Step "Asking: $question"

$bodyObj = @{ message = $question; session_id = $sessionId }
$json     = $bodyObj | ConvertTo-Json -Compress

try {
    $chatResp = Invoke-RestMethod -Uri "$BaseUrl/api/chat" -Method POST `
                                  -Headers $headers -ContentType 'application/json' `
                                  -Body $json -TimeoutSec $ChatTimeout
} catch {
    Write-Err2 "Chat request failed: $($_.Exception.Message)"
    exit 1
}

# ---- Step 4: Show the response ----
Write-Host ""
Write-Ok "===== Response ====="
$chatResp | ConvertTo-Json -Depth 6
Write-Host ""
Write-Step "Session ID: $sessionId"
Write-Info "Tip: open $BaseUrl/docs for the full interactive API documentation."
Write-Info "Tip: open $BaseUrl/ (web UI) if the frontend service is enabled."
