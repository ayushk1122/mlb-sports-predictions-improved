$python = "C:\Users\Ayush\AppData\Local\Programs\Python\Python313\python.exe"
$script = ".\modeling\evaluate_prediction_accuracy.py"

$start = Get-Date "2024-08-02"
$end = Get-Date "2024-10-02"

$total_games = 0
$total_correct = 0
$brier_scores = @()
$total_brier_weighted = 0
$all_reliability_scores = @()
$all_resolution_scores = @()
$all_uncertainty_scores = @()
$all_probability_ranges = @()
$date_details = @()

$current = $start
while ($current -le $end) {
    $dateStr = $current.ToString("yyyy-MM-dd")
    Write-Host "Evaluating $dateStr"
    $output = & $python $script --date $dateStr

    $games = $null
    $correct = $null
    $brier_score = $null
    $reliability = $null
    $resolution = $null
    $uncertainty = $null
    $foundCorrect = $false
    $foundBrier = $false
    $foundComponents = $false
    $foundBreakdown = $false
    $breakdown_lines = @()

    foreach ($line in $output) {
        if ($line -match "Total Games Evaluated: (\d+)") {
            $games = [int]$matches[1]
        }
        if (-not $foundCorrect -and $line -match "Correct Predictions: (\d+)") {
            $correct = [int]$matches[1]
            $foundCorrect = $true
        }
        if (-not $foundBrier -and $line -match "Brier Score: ([\d\.]+)") {
            $brier_score = [double]$matches[1]
            $foundBrier = $true
        }
        if (-not $foundComponents -and $line -match "Reliability \(Calibration\): ([\d\.]+)") {
            $reliability = [double]$matches[1]
        }
        if ($reliability -ne $null -and $line -match "Resolution: ([\d\.]+)") {
            $resolution = [double]$matches[1]
        }
        if ($resolution -ne $null -and $line -match "Uncertainty: ([\d\.]+)") {
            $uncertainty = [double]$matches[1]
            $foundComponents = $true
        }
        if ($line -match "Brier Score Breakdown by Probability Range:") {
            $foundBreakdown = $true
            continue
        }
        if ($foundBreakdown -and $line -match "^\s*(\d+\.\d+-\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+)\s+(\d+\.\d+)") {
            $breakdown_lines += [PSCustomObject]@{
                ProbRange = $matches[1]
                AvgPredicted = [double]$matches[2]
                AvgActual = [double]$matches[3]
                CalibrationError = [double]$matches[4]
                BrierScore = [double]$matches[5]
                Count = [int]$matches[6]
                Percentage = [double]$matches[7]
            }
        }
        if ($foundBreakdown -and $line -match "Incorrect Predictions:") {
            $foundBreakdown = $false
        }
    }

    if ($games -ne $null -and $correct -ne $null) {
        $total_games += $games
        $total_correct += $correct
        
        if ($brier_score -ne $null) {
            $brier_scores += $brier_score
            $total_brier_weighted += ($brier_score * $games)
            
            if ($reliability -ne $null) {
                $all_reliability_scores += $reliability
                $all_resolution_scores += $resolution
                $all_uncertainty_scores += $uncertainty
            }
            
            # Store date details
            $date_details += [PSCustomObject]@{
                Date = $dateStr
                Games = $games
                Correct = $correct
                Accuracy = [math]::Round(100 * $correct / $games, 2)
                BrierScore = $brier_score
                Reliability = $reliability
                Resolution = $resolution
                Uncertainty = $uncertainty
                Breakdown = $breakdown_lines
            }
            
            Write-Host "  Games: $games, Correct: $correct, Brier: $brier_score"
        } else {
            Write-Host "  Games: $games, Correct: $correct, Brier: N/A"
        }
    }

    $current = $current.AddDays(1)
}

Write-Host ""
Write-Host "=== COMPREHENSIVE CUMULATIVE RESULTS ==="
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
        
        Write-Host ""
        Write-Host "=== BRIER SCORE ANALYSIS ==="
        Write-Host "Number of dates with Brier scores: $($brier_scores.Count)"
        Write-Host "Simple Average Brier Score: $avg_brier"
        Write-Host "Weighted Average Brier Score: $weighted_avg_brier (weighted by games per date)"
        Write-Host "Brier Score Range: $min_brier to $max_brier"
        
        # Brier score components analysis
        if ($all_reliability_scores.Count -gt 0) {
            $avg_reliability = [math]::Round(($all_reliability_scores | Measure-Object -Average).Average, 4)
            $avg_resolution = [math]::Round(($all_resolution_scores | Measure-Object -Average).Average, 4)
            $avg_uncertainty = [math]::Round(($all_uncertainty_scores | Measure-Object -Average).Average, 4)
            
            Write-Host ""
            Write-Host "=== BRIER SCORE COMPONENTS (Average) ==="
            Write-Host "Reliability (Calibration): $avg_reliability"
            Write-Host "Resolution: $avg_resolution"
            Write-Host "Uncertainty: $avg_uncertainty"
        }
        
        # Aggregate probability range analysis
        Write-Host ""
        Write-Host "=== AGGREGATE PROBABILITY RANGE ANALYSIS ==="
        $all_ranges = @{}
        
        foreach ($date in $date_details) {
            if ($date.Breakdown) {
                foreach ($range in $date.Breakdown) {
                    $key = $range.ProbRange
                    if (-not $all_ranges.ContainsKey($key)) {
                        $all_ranges[$key] = @{
                            TotalCount = 0
                            TotalPredicted = 0
                            TotalActual = 0
                            TotalBrier = 0
                            Dates = 0
                        }
                    }
                    $all_ranges[$key].TotalCount += $range.Count
                    $all_ranges[$key].TotalPredicted += ($range.AvgPredicted * $range.Count)
                    $all_ranges[$key].TotalActual += ($range.AvgActual * $range.Count)
                    $all_ranges[$key].TotalBrier += ($range.BrierScore * $range.Count)
                    $all_ranges[$key].Dates += 1
                }
            }
        }
        
        # Calculate aggregate metrics for each range
        $aggregate_ranges = @()
        foreach ($range in $all_ranges.Keys | Sort-Object) {
            $data = $all_ranges[$range]
            if ($data.TotalCount -gt 0) {
                $avg_predicted = [math]::Round($data.TotalPredicted / $data.TotalCount, 3)
                $avg_actual = [math]::Round($data.TotalActual / $data.TotalCount, 3)
                $calibration_error = [math]::Round([math]::Abs($avg_predicted - $avg_actual), 3)
                $avg_brier = [math]::Round($data.TotalBrier / $data.TotalCount, 4)
                $percentage = [math]::Round(100 * $data.TotalCount / $total_games, 1)
                
                $aggregate_ranges += [PSCustomObject]@{
                    ProbRange = $range
                    AvgPredicted = $avg_predicted
                    AvgActual = $avg_actual
                    CalibrationError = $calibration_error
                    BrierScore = $avg_brier
                    TotalCount = $data.TotalCount
                    Percentage = $percentage
                    Dates = $data.Dates
                }
            }
        }
        
        if ($aggregate_ranges.Count -gt 0) {
            Write-Host "Aggregate Brier Score Breakdown by Probability Range:"
            $aggregate_ranges | Format-Table -AutoSize
        }
        
        # Performance consistency analysis
        Write-Host ""
        Write-Host "=== PERFORMANCE CONSISTENCY ANALYSIS ==="
        $accuracies = $date_details | ForEach-Object { $_.Accuracy }
        $avg_accuracy = [math]::Round(($accuracies | Measure-Object -Average).Average, 2)
        $min_accuracy = [math]::Round(($accuracies | Measure-Object -Minimum).Minimum, 2)
        $max_accuracy = [math]::Round(($accuracies | Measure-Object -Maximum).Maximum, 2)
        $accuracy_std = [math]::Round((($accuracies | ForEach-Object { ($_ - $avg_accuracy) * ($_ - $avg_accuracy) } | Measure-Object -Average).Average), 2)
        
        Write-Host "Accuracy Statistics:"
        Write-Host "  Average: $avg_accuracy%"
        Write-Host "  Range: $min_accuracy% to $max_accuracy%"
        Write-Host "  Standard Deviation: $accuracy_std%"
        
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
        
        # Save detailed results to CSV
        $output_path = "comprehensive_evaluation_results.csv"
        $date_details | Export-Csv -Path $output_path -NoTypeInformation
        Write-Host ""
        Write-Host "Detailed results saved to: $output_path"
        
    } else {
        Write-Host "No Brier scores available for analysis"
    }
} else {
    Write-Host "No games evaluated."
}