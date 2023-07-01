# This file is part of the wen package.

latex_env_pair = {
    r"\(": r"\)",
    r"\[": r"\]",
    r"\begin": r"\end",
}

user_define_map = dict(
    [
        ("8alpha", "α"),
        ("8beta", "β"),
        ("8gamma", "γ"),
        ("8Delta", "Δ"),
        ("9/check", "- [ ] "),
    ]
)


def latex_open_left(text):
    for left, right in latex_env_pair.items():
        left_idx = text.rfind(left)
        if left_idx >= 0:
            right_idx = text.rfind(right)
            if right_idx < left_idx:
                return left


def latex_open_right(text, left, right):
    right_idx = text.find(right)
    if right_idx >= 0:
        left_idx = text.find(left)
        if left_idx < 0 or right_idx < left_idx:
            return right


def in_latex_env(doc, pos):
    offset = doc.offset_at_position(pos)
    text_before = doc.source[:offset]
    text_after = doc.source[offset:]
    left = latex_open_left(text_before)
    if left:
        right = latex_env_pair[left]
        return latex_open_right(text_after, left, right) is not None
    return False
