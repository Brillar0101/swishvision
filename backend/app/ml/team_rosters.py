"""
NBA Team Rosters for Player Identification.
Maps jersey numbers to player names for each team.
"""

# 2024-25 NBA Finals: Indiana Pacers vs Oklahoma City Thunder
TEAM_ROSTERS = {
    "Indiana Pacers": {
        "0": "Haliburton",
        "1": "Toppin",
        "2": "Nembhard",
        "5": "Walker",
        "8": "Freeman",
        "9": "McConnell",
        "00": "Mathurin",
        "12": "Furphy",
        "16": "Johnson",
        "22": "I. Jackson",
        "23": "Nesmith",
        "26": "Sheppard",
        "29": "Q. Jackson",
        "33": "Turner",
        "43": "Siakam",
    },
    "Oklahoma City Thunder": {
        "2": "Gilgeous-Alexander",
        "3": "D. Jones",
        "5": "Dort",
        "6": "Jaylin Williams",
        "7": "Holmgren",
        "8": "Jalen Williams",
        "9": "Caruso",
        "11": "Joe",
        "13": "Dieng",
        "14": "Flagler",
        "21": "Wiggins",
        "22": "Wallace",
        "25": "Mitchell",
        "34": "Kenrich Williams",
        "55": "Hartenstein",
    }
}

TEAM_COLORS = {
    "Indiana Pacers": "#FDBB30",      # Yellow/Gold
    "Oklahoma City Thunder": "#FFFFFF" # White
}

# Team jersey color descriptors for matching
TEAM_JERSEY_COLORS = {
    "Indiana Pacers": "yellow",
    "Oklahoma City Thunder": "white"
}


def get_player_name(team_name: str, jersey_number: str) -> str:
    """
    Get player name from team and jersey number.

    Args:
        team_name: Name of the team
        jersey_number: Jersey number as string

    Returns:
        Player name or None if not found
    """
    if team_name in TEAM_ROSTERS:
        return TEAM_ROSTERS[team_name].get(jersey_number)
    return None


def get_team_roster(team_name: str) -> dict:
    """
    Get full roster for a team.

    Args:
        team_name: Name of the team

    Returns:
        Dict mapping jersey numbers to player names
    """
    return TEAM_ROSTERS.get(team_name, {})
