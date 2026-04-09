param(
    [Parameter(Mandatory = $true)]
    [string]$QueryId,

    [string]$BaseUrl = "http://127.0.0.1:8000"
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
        [string]$Uri
    )

    $client = [System.Net.Http.HttpClient]::new()

    try {
        if ($Method -eq "GET") {
            $response = $client.GetAsync($Uri).GetAwaiter().GetResult()
        }
        else {
            throw "Unsupported method: $Method"
        }

        $response.EnsureSuccessStatusCode()
        $bytes = $response.Content.ReadAsByteArrayAsync().GetAwaiter().GetResult()
        $text = $utf8.GetString($bytes)
        return $text | ConvertFrom-Json
    }
    finally {
        $client.Dispose()
    }
}

$response = Invoke-Utf8JsonRequest `
    -Method "GET" `
    -Uri "$BaseUrl/api/sources/$QueryId"

$response | ConvertTo-Json -Depth 8
