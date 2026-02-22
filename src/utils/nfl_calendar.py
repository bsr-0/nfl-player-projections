"""
Single source of truth for current NFL season and week.

Computes season and week from today so app, data loading, and training
all use the same notion of "current" without hardcoded years.
"""
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple


def get_current_nfl_season(today: Optional[datetime] = None) -> int:
    """
    Current NFL season year (year when Week 1 started).

    NFL season runs Sept-Feb:
    - Jan-Aug: previous calendar year (e.g. Jan 2026 = 2025 season)
    - Sept-Dec: current calendar year (e.g. Sept 2025 = 2025 season)
    """
    if today is None:
        today = datetime.now()
    if today.month <= 8:  # Jan-Aug
        return today.year - 1
    return today.year


def _season_start(season_year: int) -> datetime:
    """Approximate Week 1 Thursday (first week of Sept)."""
    # NFL typically starts first Thursday of September
    sept = datetime(season_year, 9, 1)
    # First Thursday
    weekday = sept.weekday()  # 0=Mon, 3=Thu
    days_until_thu = (3 - weekday) % 7
    if days_until_thu == 0 and sept.day == 1 and sept.weekday() != 3:
        days_until_thu = 7
    return sept + timedelta(days=days_until_thu)


def _playoff_dates(season_year: int) -> Dict[str, datetime]:
    """Approximate playoff dates (season runs into season_year+1)."""
    next_jan = season_year + 1
    # Wild Card ~2nd weekend Jan; Divisional ~3rd; Conf Champ ~4th; SB ~1st Sun Feb
    return {
        "wild_card_start": datetime(next_jan, 1, 10),
        "wild_card_end": datetime(next_jan, 1, 12),
        "divisional_start": datetime(next_jan, 1, 17),
        "divisional_end": datetime(next_jan, 1, 18),
        "conf_champ_start": datetime(next_jan, 1, 19),
        "conf_champ_end": datetime(next_jan, 1, 26),
        "super_bowl": datetime(next_jan, 2, 8),  # First Sunday in Feb typically
    }


def get_current_nfl_week(today: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Current NFL week from today (dynamic, no hardcoded year).

    Returns dict with: week, week_num, season, is_playoffs, is_super_bowl,
    game_date, description.
    """
    if today is None:
        today = datetime.now()
    season = get_current_nfl_season(today)
    season_start = _season_start(season)
    playoffs = _playoff_dates(season)

    # Super Bowl week (after Conf Champ through Super Bowl day)
    sb = playoffs["super_bowl"]
    if datetime(season + 1, 1, 27) <= today <= sb + timedelta(days=1):
        sb_number = season - 1966  # SB I = 1967 season
        return {
            "week": f"Super Bowl {_roman(sb_number)}",
            "week_num": 22,
            "season": season,
            "is_playoffs": True,
            "is_super_bowl": True,
            "game_date": sb,
            "description": f"Super Bowl",
        }

    # Conference Championships
    if playoffs["conf_champ_start"] <= today <= playoffs["conf_champ_end"]:
        return {
            "week": "Conference Championships",
            "week_num": 21,
            "season": season,
            "is_playoffs": True,
            "is_super_bowl": False,
            "game_date": playoffs["conf_champ_end"],
            "description": "AFC & NFC Championship Games",
        }

    # Divisional Round
    if playoffs["divisional_start"] <= today <= playoffs["divisional_end"] + timedelta(days=1):
        return {
            "week": "Divisional Round",
            "week_num": 20,
            "season": season,
            "is_playoffs": True,
            "is_super_bowl": False,
            "game_date": playoffs["divisional_start"],
            "description": "Divisional Playoff Games",
        }

    # Wild Card
    if playoffs["wild_card_start"] <= today <= playoffs["wild_card_end"] + timedelta(days=1):
        return {
            "week": "Wild Card",
            "week_num": 19,
            "season": season,
            "is_playoffs": True,
            "is_super_bowl": False,
            "game_date": playoffs["wild_card_start"],
            "description": "Wild Card Playoff Games",
        }

    # Preseason
    if today < season_start:
        return {
            "week": "Preseason",
            "week_num": 0,
            "season": season,
            "is_playoffs": False,
            "is_super_bowl": False,
            "game_date": season_start,
            "description": f"Season starts {season_start.strftime('%b %d, %Y')}",
        }

    # Regular season week 1-18
    days_since_start = (today - season_start).days
    week_num = min(max(1, (days_since_start // 7) + 1), 18)
    week_start = season_start + timedelta(weeks=week_num - 1)

    return {
        "week": f"Week {week_num}",
        "week_num": week_num,
        "season": season,
        "is_playoffs": False,
        "is_super_bowl": False,
        "game_date": week_start,
        "description": f"Regular Season Week {week_num}",
    }


def get_week_label(week_num: int, season: Optional[int] = None) -> str:
    """
    Human-readable label for a week number.

    week_num: 1-18 regular season, 19-22 playoffs, 0 preseason.
    If season is provided, Super Bowl label includes roman numeral.
    """
    if week_num <= 0:
        return "Preseason"
    if week_num == 19:
        return "Wild Card"
    if week_num == 20:
        return "Divisional Round"
    if week_num == 21:
        return "Conference Championships"
    if week_num == 22:
        if season is not None:
            sb_number = season - 1966  # SB I = 1967 season
            return f"Super Bowl {_roman(sb_number)}"
        return "Super Bowl"
    return f"Week {week_num}"


def get_next_n_nfl_weeks(today: Optional[datetime] = None, n: int = 1) -> List[Tuple[int, int]]:
    """
    Return the next n NFL weeks as (season, week_num), starting from the next week to be played.
    After the Super Bowl has been played (or during SB week), the next week is next season week 1
    so schedule availability reflects whether the upcoming season's schedule has been released.

    Preseason (week_num 0) is treated as "next week = week 1 of current season".
    """
    if n <= 0:
        return []
    t = today if today is not None else datetime.now()
    info = get_current_nfl_week(t)
    season = info["season"]
    week_num = info.get("week_num", 1)
    if week_num == 0:
        week_num = 1  # first week ahead is week 1 of current season
    # After the Super Bowl has been played, or during SB week, next games are next season week 1.
    # (Dates after SB fall through to "regular season" week 18 in get_current_nfl_week, so we
    # also check the calendar: if today is past SB date, we're in offseason.)
    sb_date = _playoff_dates(season)["super_bowl"]
    sb_end = sb_date + timedelta(days=1)
    if t > sb_end or week_num == 22:
        season += 1
        week_num = 1
    out: List[Tuple[int, int]] = []
    for _ in range(n):
        out.append((season, week_num))
        if week_num < 22:
            week_num += 1
        else:
            season += 1
            week_num = 1
    return out


def current_season_has_weeks_played(today: Optional[datetime] = None) -> bool:
    """
    True if the current NFL season has at least one week played (regular or playoffs).

    Used to decide whether current season is "in progress" and should be loaded
    from PBP and used as test season.
    """
    info = get_current_nfl_week(today)
    return (info.get("week_num", 0) >= 1)


def _roman(n: int) -> str:
    """Roman numeral for Super Bowl number (e.g. 60 -> LX)."""
    val = [1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1]
    sym = ["M", "CM", "D", "CD", "C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    r = ""
    i = 0
    while n > 0:
        for _ in range(n // val[i]):
            r += sym[i]
            n -= val[i]
        i += 1
    return r


def is_future_or_current_matchup(
    data_season: int,
    data_week: int,
    today: Optional[datetime] = None,
) -> bool:
    """
    True if (data_season, data_week) is the current or upcoming week.

    Used to decide whether to show "This Week's Edge" as the upcoming matchup
    or to show a disclaimer that data is from a past week.
    """
    if today is None:
        today = datetime.now()
    current = get_current_nfl_week(today)
    cur_season = current["season"]
    cur_week_num = current["week_num"]

    if data_season > cur_season:
        return True
    if data_season < cur_season:
        return False
    return data_week >= cur_week_num
