"""Research extensions on top of the frozen desk system (fiam_desks/). Nothing in fiam_desks/ is modified: this package only imports it.

core.py     context (panel + desk parts + extended short interest), `book()` = score -> LP book, `metrics()` = the full metric suite
si_ext.py   FINRA short interest from 2018-01 (fiam_desks/si.py starts 2020-06)
ledger.py   research ledger (every variant, every window) + the TEST-look ledger of fiam_desks
stats.py    bootstrap, deflated Sharpe, PBO (CSCV), BH-FDR, paired tests
"""
