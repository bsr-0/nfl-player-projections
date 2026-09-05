"""How this project spells a player's name.

The board is built from nflverse weekly stats, which abbreviate: "J.Allen",
"A.St. Brown". Anything that has to join to it -- a draft class, an ESPN
roster -- must spell names the same way, so the rule lives here instead of in
each caller.
"""

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
