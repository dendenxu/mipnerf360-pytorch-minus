# everybody loves colorization... right? ... right?
import os
import sys
from collections import deque
from types import NoneType
from typing import Dict, Iterable, List
from termcolor import colored
from rich.live import Live
from rich.table import Table
from rich.text import Text
from tqdm import tqdm

from lib.utils.base_utils import default_dotdict, dotdict

# fmt: off
def red(x):     return colored(x, 'red')
def green(x):   return colored(x, 'green')
def blue(x):    return colored(x, 'blue')
def yellow(x):  return colored(x, 'yellow')
def cyan(x):    return colored(x, 'cyan')
def magenta(x): return colored(x, 'magenta')
# fmt: on


def print_colorful_stacktrace():
    from rich.console import Console
    console = Console()
    console.print_exception()


def colored_rgb(fg_color, bg_color, text):
    r, g, b = fg_color
    result = f'\033[38;2;{r};{g};{b}m{text}'
    r, g, b = bg_color
    result = f'\033[48;2;{r};{g};{b}m{result}\033[0m'
    return result


def run_if_not_exists(cmd, outname):
    # whether a file exists, whether a directory has more than 3 elements
    # if (os.path.exists(outname) and os.path.isfile(outname)) or (os.path.isdir(outname) and len(os.listdir(outname)) >= 3):
    if os.path.exists(outname):
        log(f'Skip: {cmd}', 'yellow')
    else:
        run(cmd)


def run(cmd, quite=False):
    if isinstance(cmd, list):
        cmd = ' '.join(list(map(str, cmd)))
    func = sys._getframe(1).f_code.co_name
    if not quite:
        cmd_color = 'blue' if not cmd.startswith('rm') else 'red'
        log(colored(func, 'yellow')+": "+colored(cmd, cmd_color))
    code = os.system(cmd)
    if code != 0:
        log(colored(str(code), 'red')+" <- "+colored(func, 'yellow')+": "+colored(cmd, 'red'))
        raise RuntimeError(f'{code} <- {func}: {cmd}')


def log(msg, color=None, attrs=None):
    func = sys._getframe(1).f_code.co_name
    frame = sys._getframe(1)
    module = frame.f_globals['__name__'] if frame is not None else ''
    tqdm.write(colored(module, 'blue') + " -> " + colored(func, 'green') + ": " + colored(str(msg), color, attrs))  # be compatible with existing tqdm loops


def create_table(name: str,
                 columns: List[str],
                 rows: List[List[str]] = [],
                 styles: default_dotdict[str, NoneType] = default_dotdict(NoneType),
                 ):
    table = Table(title=name, show_footer=True, show_header=False)
    for col in columns:
        table.add_column(footer=Text(col, styles[col]), style=styles[col], justify="center")

    for row in rows:
        table.add_row(*row)
    return table


def create_live(*args, **kwargs):
    table = create_table(*args, **kwargs)
    live = Live(table, auto_refresh=True)
    return live


def update_log_stats(states: dotdict,
                     styles: default_dotdict[str, NoneType] = default_dotdict(
                         NoneType,
                         {
                             'epoch': 'cyan',

                             'img_loss': 'magenta',
                             'psnr': 'magenta',
                             'loss': 'magenta',

                             'data': 'blue',
                             'batch': 'blue',
                         }
                     ),
                     table_row_limit=200,
                     ):

    name = states.name
    del states.name
    keys = list(states.keys())
    values = list(map(str, states.values()))

    if not hasattr(update_log_stats, 'live'):
        update_log_stats.live = create_live(name, keys, [values], styles)
        update_log_stats.live.start()

    width, height = os.get_terminal_size()
    maxlen = max(min(height - 8, table_row_limit), 1)  # 5 would fill the terminal
    if not hasattr(update_log_stats, 'rows'):
        update_log_stats.rows = deque(maxlen=maxlen)  # save space for header and footer
    elif update_log_stats.rows.maxlen != maxlen:
        update_log_stats.rows = deque(list(update_log_stats.rows)[-maxlen+1:], maxlen=maxlen)  # save space for header and footer
    update_log_stats.rows.append(values)
    update_log_stats.live.update(create_table(name, keys, update_log_stats.rows, styles))
