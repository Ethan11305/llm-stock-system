param(
    [string]$Ticker,
    [int]$Days = 30,
    [switch]$Price,
    [switch]$StockInfo,
    [switch]$ForceStockInfo,
    [switch]$Fundamentals,
    [switch]$BalanceSheet,
    [switch]$CashFlow,
    [switch]$MonthlyRevenue,
    [switch]$Valuation,
    [switch]$Dividend,
    [switch]$News,
    [string[]]$NewsKeyword,
    [switch]$Margin
)

$arguments = @("-m", "llm_stock_system.workers.sync_market_data")

if ($Ticker) {
    $arguments += @("--ticker", $Ticker)
    $arguments += @("--days", "$Days")
}

if ($Price) {
    $arguments += "--price"
}

if ($StockInfo) {
    $arguments += "--stock-info"
}

if ($ForceStockInfo) {
    $arguments += "--force-stock-info"
}

if ($Fundamentals) {
    $arguments += "--fundamentals"
}

if ($BalanceSheet) {
    $arguments += "--balance-sheet"
}

if ($CashFlow) {
    $arguments += "--cash-flow"
}

if ($MonthlyRevenue) {
    $arguments += "--monthly-revenue"
}

if ($Valuation) {
    $arguments += "--valuation"
}

if ($Dividend) {
    $arguments += "--dividend"
}

if ($News) {
    $arguments += "--news"
}

if ($NewsKeyword) {
    foreach ($keyword in $NewsKeyword) {
        if ($keyword) {
            $arguments += @("--news-keyword", $keyword)
        }
    }
}

if ($Margin) {
    $arguments += "--margin"
}

python @arguments
