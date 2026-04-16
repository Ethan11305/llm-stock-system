param(
    [Parameter(Mandatory = $true)]
    [string]$Query,

    [string]$Ticker,

    [ValidateSet("news", "earnings", "announcement", "composite")]
    [string]$Topic,

    [string]$TimeRange,

    [string]$BaseUrl = "http://127.0.0.1:8000",

    [int]$TimeoutSec = 180,

    [switch]$Raw
)

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Net.Http

$utf8 = [System.Text.UTF8Encoding]::new($false)
$OutputEncoding = [Console]::OutputEncoding = $utf8
[Console]::InputEncoding = $utf8

function Invoke-Utf8JsonRequest {
    param(
        [Parameter(Mandatory = $true)]
        [ValidateSet("GET", "POST")]
        [string]$Method,

        [Parameter(Mandatory = $true)]
        [string]$Uri,

        [hashtable]$Body,

        [int]$TimeoutSec = 180
    )

    $client = [System.Net.Http.HttpClient]::new()
    $client.Timeout = [TimeSpan]::FromSeconds($TimeoutSec)

    try {
        if ($Method -eq "POST") {
            $json = if ($null -ne $Body) {
                $Body | ConvertTo-Json -Depth 8 -Compress
            }
            else {
                "{}"
            }

            $content = [System.Net.Http.StringContent]::new(
                $json,
                $utf8,
                "application/json"
            )

            $response = $client.PostAsync($Uri, $content).GetAwaiter().GetResult()
            $content.Dispose()
        }
        else {
            $response = $client.GetAsync($Uri).GetAwaiter().GetResult()
        }

        $response.EnsureSuccessStatusCode()
        $bytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
        $text = $utf8.GetString($bytes)
        return $text | ConvertFrom-Json
    }
    catch [System.Threading.Tasks.TaskCanceledException] {
        throw "Query timed out. The API may still be processing, or the backend model/data source is responding slowly. Try a larger -TimeoutSec value, for example -TimeoutSec 300."
    }
    finally {
        $client.Dispose()
    }
}

$payload = @{
    query = $Query
}

if ($Ticker) {
    $payload.ticker = $Ticker
}

if ($Topic) {
    $payload.topic = $Topic
}

if ($TimeRange) {
    $payload.timeRange = $TimeRange
}

$response = Invoke-Utf8JsonRequest `
    -Method "POST" `
    -Uri "$BaseUrl/api/query" `
    -Body $payload `
    -TimeoutSec $TimeoutSec

if ($Raw) {
    $response | ConvertTo-Json -Depth 8
    exit 0
}

Write-Host ""
Write-Host "Query ID: $($response.query_id)"
Write-Host "Confidence: $($response.confidenceLight) ($($response.confidenceScore))"
Write-Host "Summary: $($response.summary)"
Write-Host ""
Write-Host "Highlights:"
foreach ($item in $response.highlights) {
    Write-Host "- $item"
}

Write-Host ""
Write-Host "Risks:"
foreach ($item in $response.risks) {
    Write-Host "- $item"
}

# --- Forecast block (only for forecast queries) ---
if ($null -ne $response.forecast) {
    $fc = $response.forecast
    Write-Host ""
    Write-Host "=== Forecast ==="
    Write-Host "Forecast Window: $($fc.forecast_window.label) ($($fc.forecast_window.start_date) ~ $($fc.forecast_window.end_date))"
    Write-Host "Direction: $($fc.direction)"
    if ($null -ne $fc.scenario_range) {
        Write-Host "Scenario Range: $($fc.scenario_range.low) ~ $($fc.scenario_range.high) ($($fc.scenario_range.basis_type))"
    }
    if ($null -ne $fc.forecast_basis -and $fc.forecast_basis.Count -gt 0) {
        Write-Host "Forecast Basis:"
        foreach ($basis in $fc.forecast_basis) {
            Write-Host "  - $basis"
        }
    }
    Write-Host "Mode: $($fc.mode)"
}

Write-Host ""
Write-Host "Sources API:"
Write-Host "$BaseUrl/api/sources/$($response.query_id)"
Write-Host ""
Write-Host $response.disclaimer
