"""How this project spells a player's name.

The board is built from nflverse weekly stats, which abbreviate: "J.Allen",
"A.St. Brown". Anything that has to join to it -- a draft class, an ESPN
roster -- must spell names the same way, so the rule lives here instead of in
each caller.
"""

import unicodedata

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def board_name(full) -> str:
    """"Fernando Mendoza" -> "F.Mendoza".

    Suffixes are dropped ("Kenneth Walker III" -> "K.Walker"); compound
    surnames are not, which is why Amon-Ra St. Brown is on the board as
    "A.St. Brown".
    """
    parts = [w for w in str(full).split() if w]
    if len(parts) < 2:
        return str(full)
    while len(parts) > 2 and parts[-1].lower().strip(".") in SUFFIXES:
        parts.pop()
    return f"{parts[0][0]}.{' '.join(parts[1:])}"


def full_name_key(full) -> str:
    """"Travis Etienne Jr." -> "travis etienne": a join key that keeps the
    whole first name.

    board_name() collapses to an initial, so "Travis Etienne" and "Trevor
    Etienne" (or Bijan and Brian Robinson, both on ATL) become the same key
    and any (name, position) join between two sources silently hands one
    player the other's value. Use this when both sides have a full name --
    rosters.player_name and FantasyPros ADP do -- and fall back to the
    abbreviated key only where it is provably unambiguous.
    """
    parts = [w for w in str(full).split() if w]
    while len(parts) > 2 and parts[-1].lower().strip(".") in SUFFIXES:
        parts.pop()
    # Fold diacritics: rosters spell "Audric Estimé", FantasyPros "Audric Estime".
    folded = unicodedata.normalize("NFKD", " ".join(parts))
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return (folded.lower()
            .replace(".", "").replace("'", "").replace("-", " ").strip())
