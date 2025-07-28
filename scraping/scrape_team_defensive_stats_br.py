# scrape_team_defensive_stats_br.py

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
import time

# === Logging ===
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# === Paths ===
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# === Team Abbreviations for Baseball Reference URLs ===
TEAM_ABBREVIATIONS = {
    'NYY': 'NYY', 'BOS': 'BOS', 'TB': 'TBR', 'TOR': 'TOR', 'BAL': 'BAL',
    'CLE': 'CLE', 'DET': 'DET', 'KC': 'KCR', 'CWS': 'CHW', 'MIN': 'MIN',
    'HOU': 'HOU', 'SEA': 'SEA', 'TEX': 'TEX', 'LAA': 'LAA', 'OAK': 'OAK',
    'ATL': 'ATL', 'MIA': 'MIA', 'NYM': 'NYM', 'PHI': 'PHI', 'WSH': 'WSN',
    'CHC': 'CHC', 'CIN': 'CIN', 'MIL': 'MIL', 'PIT': 'PIT', 'STL': 'STL',
    'ARI': 'ARI', 'COL': 'COL', 'LAD': 'LAD', 'SD': 'SDP', 'SF': 'SFG'
}

def setup_driver():
    """Setup Chrome driver with options - following successful scrape_game_results.py pattern."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("user-agent=Mozilla/5.0")
    
    return webdriver.Chrome(options=options)

def scrape_team_pitching_stats_br(team_abbr, season=2025):
    """
    Scrape team pitching statistics from Baseball Reference.
    """
    if team_abbr not in TEAM_ABBREVIATIONS:
        logger.warning(f"Unknown team abbreviation: {team_abbr}")
        return None
    
    br_abbr = TEAM_ABBREVIATIONS[team_abbr]
    url = f"https://www.baseball-reference.com/teams/{br_abbr}/{season}.shtml"
    
    driver = setup_driver()
    
    try:
        logger.info(f"Scraping {team_abbr} pitching stats from {url}")
        driver.get(url)
        
        # Get the page source and parse with BeautifulSoup (following scrape_game_results.py pattern)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Find the team pitching table - look for table with id containing 'pitching'
        pitching_table = soup.find('table', {'id': 'team_pitching'})
        if not pitching_table:
            # Try alternative table IDs
            pitching_table = soup.find('table', {'id': 'pitching'})
        if not pitching_table:
            # Look for any table with pitching-related content
            tables = soup.find_all('table')
            for table in tables:
                if table.find('caption') and 'pitching' in table.find('caption').get_text().lower():
                    pitching_table = table
                    break
        
        if not pitching_table:
            logger.warning(f"Could not find team pitching table for {team_abbr}")
            return None
        
        # Find the "Team Totals" row - check both tbody and tfoot
        team_totals = None
        
        # First check tbody
        tbody = pitching_table.find('tbody')
        if tbody:
            rows = tbody.find_all('tr')
            for row in rows:
                name_cell = row.find('td', {'data-stat': 'name_display'})
                if name_cell and 'Team Totals' in name_cell.get_text():
                    team_totals = row
                    break
        
        # If not found in tbody, check tfoot
        if not team_totals:
            tfoot = pitching_table.find('tfoot')
            if tfoot:
                rows = tfoot.find_all('tr')
                for row in rows:
                    name_cell = row.find('td', {'data-stat': 'name_display'})
                    if name_cell and 'Team Totals' in name_cell.get_text():
                        team_totals = row
                        break
        
        if not team_totals:
            logger.warning(f"Could not find team totals for {team_abbr}")
            return None
        
        # Extract stats using data-stat attributes
        stats = {}
        
        # Map of stat names to their data-stat attributes
        stat_mapping = {
            'games': 'p_games',
            'wins': 'p_wins',
            'losses': 'p_losses',
            'saves': 'p_saves',
            'innings_pitched': 'p_ip',
            'hits_allowed': 'p_hits',
            'runs_allowed': 'p_runs',
            'earned_runs': 'p_earned_runs',
            'home_runs_allowed': 'p_home_runs',
            'walks_allowed': 'p_walks',
            'strikeouts': 'p_strikeouts',
            'era': 'p_era',
            'whip': 'p_whip',
            'hits_per_9': 'p_hits_per_nine',
            'home_runs_per_9': 'p_home_runs_per_nine',
            'walks_per_9': 'p_walks_per_nine',
            'strikeouts_per_9': 'p_strikeouts_per_nine',
            'strikeout_walk_ratio': 'p_strikeouts_per_walk'
        }
        
        # Extract stats using data-stat attributes
        for stat_name, data_stat in stat_mapping.items():
            cell = team_totals.find('td', {'data-stat': data_stat})
            if cell:
                value = cell.get_text().strip()
                # Convert to appropriate type
                if stat_name in ['era', 'whip', 'hits_per_9', 'home_runs_per_9', 'walks_per_9', 'strikeouts_per_9', 'strikeout_walk_ratio']:
                    try:
                        stats[stat_name] = float(value) if value else 0.0
                    except ValueError:
                        stats[stat_name] = 0.0
                elif stat_name == 'innings_pitched':
                    # Handle innings pitched (e.g., "162.0" or "162.1")
                    try:
                        if '.' in value:
                            parts = value.split('.')
                            stats[stat_name] = float(parts[0]) + float(parts[1]) / 3.0
                        else:
                            stats[stat_name] = float(value) if value else 0.0
                    except ValueError:
                        stats[stat_name] = 0.0
                else:
                    try:
                        stats[stat_name] = int(value) if value else 0
                    except ValueError:
                        stats[stat_name] = 0
            else:
                stats[stat_name] = 0
        
        # Calculate additional metrics
        if stats.get('innings_pitched', 0) > 0:
            # Per game metrics
            games = stats.get('games', 1)
            stats['runs_allowed_per_game'] = stats.get('runs_allowed', 0) / games
            stats['hits_allowed_per_game'] = stats.get('hits_allowed', 0) / games
            stats['walks_allowed_per_game'] = stats.get('walks_allowed', 0) / games
            stats['strikeouts_per_game'] = stats.get('strikeouts', 0) / games
            stats['home_runs_allowed_per_game'] = stats.get('home_runs_allowed', 0) / games
            
            # Win percentage
            total_decisions = stats.get('wins', 0) + stats.get('losses', 0)
            stats['win_pct'] = stats.get('wins', 0) / total_decisions if total_decisions > 0 else 0
        
        logger.info(f"Successfully scraped {len(stats)} pitching stats for {team_abbr}")
        return stats
        
    except Exception as e:
        logger.error(f"Error scraping {team_abbr}: {e}")
        return None
    finally:
        driver.quit()

def scrape_team_fielding_stats_br(team_abbr, season=2025):
    """
    Scrape team fielding statistics from Baseball Reference.
    """
    if team_abbr not in TEAM_ABBREVIATIONS:
        logger.warning(f"Unknown team abbreviation: {team_abbr}")
        return None
    
    br_abbr = TEAM_ABBREVIATIONS[team_abbr]
    url = f"https://www.baseball-reference.com/teams/{br_abbr}/{season}.shtml"
    
    driver = setup_driver()
    
    try:
        logger.info(f"Scraping {team_abbr} fielding stats from {url}")
        driver.get(url)
        
        # Get the page source and parse with BeautifulSoup (following scrape_game_results.py pattern)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Find the team fielding table - look for table with id containing 'fielding'
        fielding_table = soup.find('table', {'id': 'team_fielding'})
        if not fielding_table:
            # Try alternative table IDs
            fielding_table = soup.find('table', {'id': 'fielding'})
        if not fielding_table:
            # Look for any table with fielding-related content
            tables = soup.find_all('table')
            for table in tables:
                if table.find('caption') and 'fielding' in table.find('caption').get_text().lower():
                    fielding_table = table
                    break
        
        if not fielding_table:
            logger.warning(f"Could not find team fielding table for {team_abbr}")
            return None
        
        # Find the "Team Totals" row - check both tbody and tfoot
        team_totals = None
        
        # First check tbody
        tbody = fielding_table.find('tbody')
        if tbody:
            rows = tbody.find_all('tr')
            for row in rows:
                name_cell = row.find('td', {'data-stat': 'name_display'})
                if name_cell and 'Team Totals' in name_cell.get_text():
                    team_totals = row
                    break
        
        # If not found in tbody, check tfoot
        if not team_totals:
            tfoot = fielding_table.find('tfoot')
            if tfoot:
                rows = tfoot.find_all('tr')
                for row in rows:
                    name_cell = row.find('td', {'data-stat': 'name_display'})
                    if name_cell and 'Team Totals' in name_cell.get_text():
                        team_totals = row
                        break
        
        if not team_totals:
            logger.warning(f"Could not find team totals for {team_abbr}")
            return None
        
        # Extract stats using data-stat attributes
        stats = {}
        
        # Map of stat names to their data-stat attributes
        stat_mapping = {
            'games': 'f_games',
            'putouts': 'f_putouts',
            'assists': 'f_assists',
            'errors': 'f_errors',
            'fielding_pct': 'f_fielding_perc',
            'double_plays': 'f_double_plays',
            'range_factor_per_game': 'f_range_factor_per_game'
        }
        
        # Extract stats using data-stat attributes
        for stat_name, data_stat in stat_mapping.items():
            cell = team_totals.find('td', {'data-stat': data_stat})
            if cell:
                value = cell.get_text().strip()
                # Convert to appropriate type
                if stat_name == 'fielding_pct':
                    try:
                        # Remove leading dot if present
                        if value.startswith('.'):
                            value = '0' + value
                        stats[stat_name] = float(value) if value else 0.0
                    except ValueError:
                        stats[stat_name] = 0.0
                else:
                    try:
                        stats[stat_name] = int(value) if value else 0
                    except ValueError:
                        stats[stat_name] = 0
            else:
                stats[stat_name] = 0
        
        logger.info(f"Successfully scraped {len(stats)} fielding stats for {team_abbr}")
        return stats
        
    except Exception as e:
        logger.error(f"Error scraping {team_abbr}: {e}")
        return None
    finally:
        driver.quit()

def scrape_team_defensive_stats_br(team_abbr, season=2025):
    """
    Scrape both pitching and fielding stats for a team.
    """
    pitching_stats = scrape_team_pitching_stats_br(team_abbr, season)
    fielding_stats = scrape_team_fielding_stats_br(team_abbr, season)
    
    if pitching_stats:
        # Combine stats
        combined_stats = {**pitching_stats}
        if fielding_stats:
            combined_stats.update(fielding_stats)
        return combined_stats
    
    return None

def scrape_all_teams_defensive_stats(season=2025, target_date=None):
    """
    Scrape defensive statistics for all MLB teams.
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Scraping team defensive stats for {season} season on {target_date}")
    
    all_team_stats = []
    
    for team_abbr in TEAM_ABBREVIATIONS.keys():
        logger.info(f"Processing {team_abbr}...")
        
        stats = scrape_team_defensive_stats_br(team_abbr, season)
        if stats:
            # Add metadata
            team_data = {
                'team_name': team_abbr,
                'season': season,
                'date': target_date,
                **stats
            }
            
            all_team_stats.append(team_data)
            logger.info(f"Successfully processed {team_abbr} with {len(stats)} metrics")
        else:
            logger.warning(f"Failed to get stats for {team_abbr}")
        
        # Rate limiting
        time.sleep(2)
    
    if all_team_stats:
        # Create DataFrame
        df = pd.DataFrame(all_team_stats)
        
        # Save to file
        output_path = PROCESSED_DIR / f"team_defensive_stats_{target_date}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} team defensive stats to {output_path}")
        
        return output_path
    else:
        logger.error("No team stats collected")
        return None

def scrape_team_defensive_stats_rolling(days_back=30, season=2025, target_date=None, overwrite=False):
    """
    Scrape team defensive stats for a rolling window of days.
    """
    if target_date is None:
        base_date = datetime.now().date()
    else:
        if isinstance(target_date, str):
            base_date = datetime.strptime(target_date, '%Y-%m-%d').date()
        else:
            base_date = target_date
    
    successful_scrapes = []
    
    for i in range(days_back):
        date = base_date - timedelta(days=i)
        date_str = date.strftime('%Y-%m-%d')
        output_path = PROCESSED_DIR / f"team_defensive_stats_{date_str}.csv"
        
        if output_path.exists() and not overwrite:
            logger.info(f"Skipping {date_str} (already exists)")
            continue
            
        try:
            logger.info(f"Scraping team defensive stats for {date_str}...")
            result = scrape_all_teams_defensive_stats(season, date_str)
            if result:
                successful_scrapes.append(date_str)
                logger.info(f"Successfully scraped {date_str}")
            else:
                logger.warning(f"No data for {date_str}")
        except Exception as e:
            logger.error(f"Error scraping data for {date_str}: {e}")
    
    logger.info(f"Completed rolling scrape. Successfully scraped {len(successful_scrapes)} dates.")
    return successful_scrapes

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Scrape MLB team defensive/pitching statistics from Baseball Reference')
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--season", type=int, default=2025, help="MLB season to scrape")
    parser.add_argument("--days_back", type=int, default=30, help="How many days back to scrape for rolling scrape")
    parser.add_argument("--rolling", action="store_true", help="Perform rolling scrape")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--team", type=str, help="Scrape specific team only (e.g., SEA)")
    
    args = parser.parse_args()
    
    if args.team:
        # Test single team
        stats = scrape_team_defensive_stats_br(args.team, args.season)
        if stats:
            print(f"\n{args.team} Defensive Stats:")
            for key, value in stats.items():
                print(f"{key}: {value}")
    elif args.rolling:
        scrape_team_defensive_stats_rolling(
            days_back=args.days_back, 
            season=args.season,
            target_date=args.date, 
            overwrite=args.overwrite
        )
    else:
        scrape_all_teams_defensive_stats(args.season, args.date)

# Usage examples:
# python scrape_team_defensive_stats_br.py --team SEA --season 2025
# python scrape_team_defensive_stats_br.py --date 2025-01-15 --season 2025
# python scrape_team_defensive_stats_br.py --rolling --days_back 45 --season 2025 