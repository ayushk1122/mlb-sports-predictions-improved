$python = "C:\Users\Ayush\AppData\Local\Programs\Python\Python313\python.exe"
$script = ".\modeling\evaluate_prediction_accuracy.py"

$start = Get-Date "2024-08-02"
$end = Get-Date "2024-10-02"

$total_games = 0
$total_correct = 0
$brier_scores = @()
$total_brier_weighted = 0
$date_details = @()

$current = $start
while ($current -le $end) {
    $dateStr = $current.ToString("yyyy-MM-dd")
    Write-Host "Evaluating $dateStr"
    $output = & $python $script --date $dateStr

    $games = $null
    $correct = $null
    $brier_score = $null
    $foundCorrect = $false

    foreach ($line in $output) {
        if ($line -match "Total Games Evaluated: (\d+)") {
            $games = [int]$matches[1]
        }
        if (-not $foundCorrect -and $line -match "Correct Predictions: (\d+)") {
            $correct = [int]$matches[1]
            $foundCorrect = $true  # Only capture the first occurrence
        }
        if ($line -match "Brier Score: ([\d\.]+)") {
            $brier_score = [double]$matches[1]
        }
    }

    if ($games -ne $null -and $correct -ne $null) {
        $total_games += $games
        $total_correct += $correct
        
        if ($brier_score -ne $null) {
            $brier_scores += $brier_score
            $total_brier_weighted += ($brier_score * $games)
            
            # Store details for validation
            $date_details += [PSCustomObject]@{
                Date = $dateStr
                Games = $games
                Correct = $correct
                BrierScore = $brier_score
                WeightedBrier = $brier_score * $games
            }
            
            Write-Host "  Games: $games, Correct: $correct, Brier: $brier_score"
        } else {
            Write-Host "  Games: $games, Correct: $correct, Brier: N/A"
        }
    }

    $current = $current.AddDays(1)
}

Write-Host ""
Write-Host "=== CUMULATIVE RESULTS ==="
Write-Host "Total Games Evaluated: $total_games"
Write-Host "Total Correct Predictions: $total_correct"
if ($total_games -gt 0) {
    $accuracy = [math]::Round(100 * $total_correct / $total_games, 2)
    Write-Host "Overall Accuracy: $accuracy%"
    
    # Calculate Brier score statistics
    if ($brier_scores.Count -gt 0) {
        $avg_brier = [math]::Round(($brier_scores | Measure-Object -Average).Average, 4)
        $weighted_avg_brier = [math]::Round($total_brier_weighted / $total_games, 4)
        $min_brier = [math]::Round(($brier_scores | Measure-Object -Minimum).Minimum, 4)
        $max_brier = [math]::Round(($brier_scores | Measure-Object -Maximum).Maximum, 4)
        
        # Manual verification of weighted average
        $manual_weighted_sum = ($date_details | ForEach-Object { $_.WeightedBrier } | Measure-Object -Sum).Sum
        $manual_weighted_avg = [math]::Round($manual_weighted_sum / $total_games, 4)
        
        Write-Host ""
        Write-Host "=== BRIER SCORE ANALYSIS ==="
        Write-Host "Number of dates with Brier scores: $($brier_scores.Count)"
        Write-Host "Simple Average Brier Score: $avg_brier"
        Write-Host "Weighted Average Brier Score: $weighted_avg_brier (weighted by games per date)"
        Write-Host "Manual Verification - Weighted Sum: $manual_weighted_sum"
        Write-Host "Manual Verification - Weighted Average: $manual_weighted_avg"
        Write-Host "Brier Score Range: $min_brier to $max_brier"
        
        # Show a few sample calculations for verification
        Write-Host ""
        Write-Host "=== SAMPLE CALCULATIONS (First 5 dates) ==="
        $date_details | Select-Object -First 5 | ForEach-Object {
            Write-Host "$($_.Date): $($_.Games) games × $($_.BrierScore) = $($_.WeightedBrier)"
        }
        
        # Brier score interpretation
        Write-Host ""
        Write-Host "=== BRIER SCORE INTERPRETATION ==="
        if ($weighted_avg_brier -lt 0.20) {
            Write-Host "EXCELLENT: Professional-level probabilistic accuracy"
        } elseif ($weighted_avg_brier -lt 0.22) {
            Write-Host "VERY GOOD: Strong probabilistic accuracy, good for betting"
        } elseif ($weighted_avg_brier -lt 0.25) {
            Write-Host "GOOD: Better than random, potentially viable for betting"
        } elseif ($weighted_avg_brier -lt 0.30) {
            Write-Host "FAIR: Slightly better than random, needs improvement"
        } else {
            Write-Host "POOR: Below random guessing level, significant improvement needed"
        }
        
        Write-Host "Random guessing would score: 0.25"
        Write-Host "Perfect predictions would score: 0.00"
    } else {
        Write-Host "No Brier scores available for analysis"
    }
} else {
    Write-Host "No games evaluated."
} 