# scrape_team_offensive_stats_br.py

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

def scrape_team_offensive_stats_br(team_abbr, season=2025):
    """
    Scrape team offensive statistics from Baseball Reference.
    """
    if team_abbr not in TEAM_ABBREVIATIONS:
        logger.warning(f"Unknown team abbreviation: {team_abbr}")
        return None
    
    br_abbr = TEAM_ABBREVIATIONS[team_abbr]
    url = f"https://www.baseball-reference.com/teams/{br_abbr}/{season}.shtml"
    
    driver = setup_driver()
    
    try:
        logger.info(f"Scraping {team_abbr} offensive stats from {url}")
        driver.get(url)
        
        # Get the page source and parse with BeautifulSoup (following scrape_game_results.py pattern)
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Find the team batting table - look for table with id containing 'batting'
        batting_table = soup.find('table', {'id': 'team_batting'})
        if not batting_table:
            # Try alternative table IDs
            batting_table = soup.find('table', {'id': 'batting'})
        if not batting_table:
            # Look for any table with batting-related content
            tables = soup.find_all('table')
            for table in tables:
                if table.find('caption') and 'batting' in table.find('caption').get_text().lower():
                    batting_table = table
                    break
        
        if not batting_table:
            logger.warning(f"Could not find team batting table for {team_abbr}")
            return None
        
        # Find the "Team Totals" row - check both tbody and tfoot
        team_totals = None
        
        # First check tbody
        tbody = batting_table.find('tbody')
        if tbody:
            rows = tbody.find_all('tr')
            for row in rows:
                name_cell = row.find('td', {'data-stat': 'name_display'})
                if name_cell and 'Team Totals' in name_cell.get_text():
                    team_totals = row
                    break
        
        # If not found in tbody, check tfoot
        if not team_totals:
            tfoot = batting_table.find('tfoot')
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
            'games': 'b_games',
            'at_bats': 'b_ab',
            'runs': 'b_r',
            'hits': 'b_h',
            'doubles': 'b_doubles',
            'triples': 'b_triples',
            'home_runs': 'b_hr',
            'rbi': 'b_rbi',
            'walks': 'b_bb',
            'strikeouts': 'b_so',
            'stolen_bases': 'b_sb',
            'caught_stealing': 'b_cs',
            'batting_avg': 'b_batting_avg',
            'on_base_pct': 'b_onbase_perc',
            'slugging_pct': 'b_slugging_perc',
            'ops': 'b_onbase_plus_slugging',
            'sacrifice_flies': 'b_sf',
            'hit_by_pitch': 'b_hbp',
            'total_bases': 'b_tb',
            'ground_into_double_plays': 'b_gidp',
            'intentional_walks': 'b_ibb'
        }
        
        # Extract stats using data-stat attributes
        for stat_name, data_stat in stat_mapping.items():
            cell = team_totals.find('td', {'data-stat': data_stat})
            if cell:
                value = cell.get_text().strip()
                # Convert to appropriate type
                if stat_name in ['batting_avg', 'on_base_pct', 'slugging_pct', 'ops']:
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
        
        # Calculate additional metrics
        if stats.get('at_bats', 0) > 0:
            # ISO (Isolated Power)
            stats['iso'] = stats.get('slugging_pct', 0) - stats.get('batting_avg', 0)
            
            # Per game metrics
            games = stats.get('games', 1)
            stats['runs_per_game'] = stats.get('runs', 0) / games
            stats['hr_per_game'] = stats.get('home_runs', 0) / games
            
            # Rate metrics
            stats['k_rate'] = stats.get('strikeouts', 0) / stats.get('at_bats', 1)
            stats['bb_rate'] = stats.get('walks', 0) / stats.get('at_bats', 1)
            
            # Stolen base success rate
            sb_attempts = stats.get('stolen_bases', 0) + stats.get('caught_stealing', 0)
            stats['sb_success_rate'] = stats.get('stolen_bases', 0) / sb_attempts if sb_attempts > 0 else 0
        
        logger.info(f"Successfully scraped {len(stats)} stats for {team_abbr}")
        return stats
        
    except Exception as e:
        logger.error(f"Error scraping {team_abbr}: {e}")
        return None
    finally:
        driver.quit()

def scrape_all_teams_offensive_stats(season=2025, target_date=None):
    """
    Scrape offensive statistics for all MLB teams.
    """
    if target_date is None:
        target_date = datetime.now().strftime('%Y-%m-%d')
    
    logger.info(f"Scraping team offensive stats for {season} season on {target_date}")
    
    all_team_stats = []
    
    for team_abbr in TEAM_ABBREVIATIONS.keys():
        logger.info(f"Processing {team_abbr}...")
        
        stats = scrape_team_offensive_stats_br(team_abbr, season)
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
        output_path = PROCESSED_DIR / f"team_offensive_stats_{season}.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df)} team offensive stats to {output_path}")
        
        return output_path
    else:
        logger.error("No team stats collected")
        return None

def scrape_team_offensive_stats_rolling(days_back=30, season=2025, target_date=None, overwrite=False):
    """
    Scrape team offensive stats for a rolling window of days.
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
        output_path = PROCESSED_DIR / f"team_offensive_stats_{date_str}.csv"
        
        if output_path.exists() and not overwrite:
            logger.info(f"Skipping {date_str} (already exists)")
            continue
            
        try:
            logger.info(f"Scraping team offensive stats for {date_str}...")
            result = scrape_all_teams_offensive_stats(season, date_str)
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
    parser = argparse.ArgumentParser(description='Scrape MLB team offensive statistics from Baseball Reference')
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--season", type=int, default=2025, help="MLB season to scrape")
    parser.add_argument("--days_back", type=int, default=30, help="How many days back to scrape for rolling scrape")
    parser.add_argument("--rolling", action="store_true", help="Perform rolling scrape")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files")
    parser.add_argument("--team", type=str, help="Scrape specific team only (e.g., SEA)")
    
    args = parser.parse_args()
    
    if args.team:
        # Test single team
        stats = scrape_team_offensive_stats_br(args.team, args.season)
        if stats:
            print(f"\n{args.team} Stats:")
            for key, value in stats.items():
                print(f"{key}: {value}")
    elif args.rolling:
        scrape_team_offensive_stats_rolling(
            days_back=args.days_back, 
            season=args.season,
            target_date=args.date, 
            overwrite=args.overwrite
        )
    else:
        scrape_all_teams_offensive_stats(args.season, args.date)

# Usage examples:
# python scrape_team_offensive_stats_br.py --team SEA --season 2025
# python scrape_team_offensive_stats_br.py --date 2025-01-15 --season 2025
# python scrape_team_offensive_stats_br.py --rolling --days_back 45 --season 2025 