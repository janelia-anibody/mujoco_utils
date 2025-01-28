"""General-purpose utilities."""

from typing import Sequence


def any_substr_in_str(substrings: Sequence[str], string: str) -> bool:
    """Checks if any of substrings is in string."""
    return any(s in string for s in substrings)


def print_tree(tree: dict) -> None:
    """Prints tree view of a dict-like structure.

    Example:
    >>> some_dict = {'key': 'value',
                     'another key': 'another value',
                     'nested dict': {'hello': 42,
                                     'world': {123: '', '3.14': ''}}}
    >>> print_tree(some_dict)
    ├─── key:
    │    └─── value
    ├─── another key:
    │    └─── another value
    └─── nested dict:
         ├─── hello:
         │    └─── 42
         └─── world:
              ├─── 123
              └─── 3.14
    """
    def _tree(tree, last):
        def get_str(last):
            return ''.join(["\u2502    " if s else "     " for s in last[:-1]])
        if not hasattr(tree, 'keys'):
            last.append(False)
            if str(tree):
                print(get_str(last) + '\u2514\u2500\u2500\u2500 ' + str(tree))
            pop = last.pop()
            while not pop and len(last):
                pop = last.pop()
        else:
            for key in tree.keys():
                last.append(False if key == list(tree.keys())[-1] else True)
                cross = '\u251c' if last[-1] else '\u2514'
                print(get_str(last) + cross + '\u2500\u2500\u2500 ' + str(key))
                _tree(tree[key], last)
    _tree(tree, last=[])
