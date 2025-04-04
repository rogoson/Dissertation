import os
from IPython.core.display import display_html
from tabulate import tabulate


def pathJoin(firstStr: str, secondStr: str):
    return os.path.join(str(firstStr), secondStr)


"""
The below is robbed from CM50268 Coursework 1 & 2 Setup Code
"""


def tabulate_neatly(table, headers=None, title=None, **kwargs):
    # Example Usage:
    # table = [["Column 1","Column 2"]]
    # table.append([Column_1_Value, Column_2_Value])
    # setup.tabulate_neatly(table, headers="firstrow", title="Table Title")
    headers = headers or "keys"
    if title is not None:
        display_html(f"<h3>{title}</h3>\n", raw=True)
    display_html(tabulate(table, headers=headers, tablefmt="html", **kwargs))
