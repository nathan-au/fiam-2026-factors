# Factor Guide — the 147 Stock Characteristics

Plain-English guide to every predictor column in `chars_final_with_names.parquet` (the `variable` list in `factor_char_list.csv`). Formal names and research references are in Appendix A of the challenge PDF. The names follow the Jensen–Kelly–Pedersen (2022) "Global Factor Data" conventions, which the PDF table only partly covers.

Every value is a snapshot of one stock at the end of one month. The model's job is to learn which snapshots come before good or bad returns **next** month.

---

## Contents

- [Quick glossary](#quick-glossary)
- [How to read the column names](#how-to-read-the-column-names)
- [1. Size & Firm Basics](#1-size--firm-basics-3)
- [2. Value (Cheap vs Expensive)](#2-value-cheap-vs-expensive-16)
- [3. Share & Debt Issuance](#3-share--debt-issuance-5)
- [4. Momentum](#4-momentum-8)
- [5. Reversal](#5-reversal-2)
- [6. Seasonality](#6-seasonality-4)
- [7. Profitability](#7-profitability-18)
- [8. Growth & Earnings Surprises](#8-growth--earnings-surprises-15)
- [9. Investment & Asset Growth](#9-investment--asset-growth-14)
- [10. Accruals & Balance-Sheet Changes](#10-accruals--balance-sheet-changes-16)
- [11. Leverage & Financial Health](#11-leverage--financial-health-8)
- [12. Earnings Stability](#12-earnings-stability-4)
- [13. Composite Quality Scores](#13-composite-quality-scores-7)
- [14. Market Risk (Beta)](#14-market-risk-beta-6)
- [15. Volatility & Lottery-Like Behaviour](#15-volatility--lottery-like-behaviour-12)
- [16. Liquidity & Trading Activity](#16-liquidity--trading-activity-9)

---

## Quick glossary

| Term | Meaning |
|---|---|
| **Market cap** (market equity) | What the whole company is worth on the stock market: share price × number of shares. |
| **Book value / book equity** | What the company is worth on its own accounting books: assets minus debts. |
| **Assets** | Everything the company owns: cash, buildings, inventory, money owed to it. |
| **Enterprise value** | Company value including its debt, i.e. roughly what it would cost to buy the whole business. |
| **Sales / revenue** | Money coming in from customers, before costs. |
| **Gross profit** | Sales minus the direct cost of making the product. |
| **Operating profit / EBIT** | Profit from running the business, before interest and taxes. |
| **EBITDA** | Operating profit before also subtracting depreciation (the accounting wear-and-tear of equipment). |
| **Net income / earnings** | The bottom-line profit after every cost, interest payment and tax. |
| **Cash flow** | Actual cash that came in or went out, as opposed to accounting profit. |
| **Accruals** | The gap between reported profit and actual cash. Big gaps can be a warning sign. |
| **Capex** | Capital expenditure: money spent on long-lived things like factories and equipment. |
| **SG&A** | Selling, general & administrative costs: overhead such as salaries, rent and marketing. |
| **Beta** | How much a stock tends to move when the overall market moves (1 = same, 2 = twice as much). |
| **Idiosyncratic** | The part of a stock's movement that the overall market or common factors don't explain; it's specific to that company. |
| **Skewness** | Whether a stock's big moves are mostly up-jumps (positive) or crashes (negative). |
| **CAPM / FF3 / q-factor (HXZ4)** | Standard models that explain returns with the market (CAPM), market + size + value (FF3), or market + size + investment + profitability (q-factor). |

## How to read the column names

| Pattern | Meaning | Example |
|---|---|---|
| `x_me` | x divided by market cap | `ni_me` = earnings ÷ market cap |
| `x_at` | x divided by total assets | `gp_at` = gross profit ÷ assets |
| `x_be` | x divided by book equity | `ni_be` = return on equity |
| `x_bev` / `x_mev` | x divided by book / market enterprise value | `ebitda_mev` |
| `..._atl1`, `..._bel1` | Uses **last year's** assets / book equity as the divisor | `gp_atl1` |
| `_gr1`, `_gr3` | Percentage growth over 1 or 3 years | `sale_gr3` |
| `_gr1a` | 1-year change scaled by total assets | `inv_gr1a` |
| `_chg1` | Change since the same point last year | `niq_at_chg1` |
| `q` in the name (`niq`, `saleq`) | Quarterly figure instead of annual | `niq_at` |
| `_21d`, `_126d`, `_252d`, `_1260d` | Computed from roughly 1 month, 6 months, 1 year or 5 years of daily data | `rvol_21d` |
| `_60m` | Computed from 60 months of monthly data | `beta_60m` |
| `ret_a_b` | Return from *a* months ago up to *b* months ago | `ret_12_1` |

---

## 1. Size & Firm Basics (3)

How big, how old, and how expensive a single share is.

| Code | Name | What it means |
|---|---|---|
| `market_equity` | Market equity | The company's total stock-market value (market cap), in millions of dollars. |
| `prc` | Share price | The price of one share at month end. |
| `age` | Firm age | How many months the company has appeared in the financial databases, a rough stand-in for how established it is. |

## 2. Value (Cheap vs Expensive) (16)

Compares what the market pays for the stock to what the business actually has or produces. A high ratio usually means the stock is "cheap".

| Code | Name | What it means |
|---|---|---|
| `be_me` | Book-to-market | Accounting value of the company divided by its market value; high means the stock looks cheap relative to its books. |
| `at_me` | Assets-to-market | Total assets the company owns divided by its market value. |
| `bev_mev` | Book-to-market enterprise value | Book value of the whole business (including debt) divided by its market value (including debt). |
| `sale_me` | Sales-to-price | Yearly revenue divided by market cap: how many dollars of sales each dollar of stock buys you. |
| `ni_me` | Earnings-to-price | Yearly profit divided by market cap, the inverse of the familiar P/E ratio. |
| `ocf_me` | Operating cash flow-to-price | Cash generated by day-to-day operations divided by market cap. |
| `fcf_me` | Free cash flow-to-price | Cash left over after operations and equipment spending, divided by market cap. |
| `ebitda_mev` | EBITDA-to-enterprise value | Operating earnings before depreciation divided by the value of the whole business including debt. |
| `div12m_me` | Dividend yield | Dividends paid to shareholders over the last 12 months divided by market cap. |
| `eqpo_me` | Payout yield | Dividends plus share buybacks divided by market cap: total cash handed back to shareholders. |
| `eqnpo_me` | Net payout yield | Dividends plus buybacks minus new shares sold, divided by market cap. |
| `debt_me` | Debt-to-market | Total debt divided by market cap. |
| `netdebt_me` | Net debt-to-price | Debt minus cash on hand, divided by market cap. |
| `rd_me` | R&D-to-market | Research & development spending divided by market cap. |
| `intrinsic_value` | Intrinsic value-to-market | An accounting-model estimate of what the company is "really" worth, compared with its market price. |
| `eq_dur` | Equity duration | How far into the future the company's cash flows are expected to arrive; high means a long-dated, growth-style stock. |

## 3. Share & Debt Issuance (5)

Whether the company is raising money (selling shares, borrowing) or returning it (buybacks, paying down debt). Firms that issue a lot tend to underperform.

| Code | Name | What it means |
|---|---|---|
| `chcsho_12m` | Net stock issues | Percentage change in the number of shares outstanding over the last year. |
| `eqnpo_12m` | Equity net payout | How much value flowed back to shareholders over 12 months via buybacks and dividends, net of new share sales. |
| `netis_at` | Net total issuance | New money raised from shares and debt combined, divided by total assets. |
| `eqnetis_at` | Net equity issuance | New money raised by selling shares (minus buybacks), divided by total assets. |
| `dbnetis_at` | Net debt issuance | New borrowing minus debt repaid, divided by total assets. |

## 4. Momentum (8)

Past winners often keep winning for a while, and past losers keep losing.

| Code | Name | What it means |
|---|---|---|
| `ret_3_1` | 3-month momentum | Stock return from 3 months ago up to 1 month ago. |
| `ret_6_1` | 6-month momentum | Stock return from 6 months ago up to 1 month ago. |
| `ret_9_1` | 9-month momentum | Stock return from 9 months ago up to 1 month ago. |
| `ret_12_1` | 12-month momentum | Stock return from 12 months ago up to 1 month ago, the classic momentum signal. |
| `ret_12_7` | Intermediate momentum | Stock return from 12 months ago up to 7 months ago, ignoring the most recent half-year. |
| `resff3_6_1` | 6-month residual momentum | Past 6-month return after removing the part explained by the overall market, size and value trends. |
| `resff3_12_1` | 12-month residual momentum | Past 12-month return after removing the part explained by the overall market, size and value trends. |
| `prc_highprc_252d` | Price vs 52-week high | Today's price divided by the highest price of the past year; near 1 means it's trading close to its peak. |

## 5. Reversal (2)

Over very short or very long horizons, returns tend to snap back the other way.

| Code | Name | What it means |
|---|---|---|
| `ret_1_0` | Short-term reversal | The stock's return in the most recent month; big recent moves often partly undo themselves. |
| `ret_60_12` | Long-term reversal | Stock return from 5 years ago up to 1 year ago; long-run big winners tend to cool off. |

## 6. Seasonality (4)

Some stocks tend to do well or poorly in the same calendar months each year.

| Code | Name | What it means |
|---|---|---|
| `seas_1_1an` | Same month, last year | The stock's return in this same calendar month one year ago. |
| `seas_1_1na` | Other months, last year | The stock's average return in the *other* 11 months of last year. |
| `seas_2_5an` | Same month, years 2–5 | The stock's average return in this same calendar month, 2 to 5 years ago. |
| `seas_2_5na` | Other months, years 2–5 | The stock's average return in all the *other* calendar months, 2 to 5 years ago. |

## 7. Profitability (18)

How good the company is at turning its resources into profit. More profitable firms have historically earned higher returns.

| Code | Name | What it means |
|---|---|---|
| `gp_at` | Gross profits-to-assets | Sales minus direct production costs, divided by total assets. |
| `gp_atl1` | Gross profits-to-lagged assets | Same as `gp_at` but divided by last year's assets. |
| `op_at` | Operating profits-to-assets | Profit from core operations divided by total assets. |
| `op_atl1` | Operating profits-to-lagged assets | Same as `op_at` but divided by last year's assets. |
| `cop_at` | Cash-based operating profits-to-assets | Operating profit adjusted to count only cash actually received, divided by total assets. |
| `cop_atl1` | Cash-based operating profits-to-lagged assets | Same as `cop_at` but divided by last year's assets. |
| `ope_be` | Operating profits-to-book equity | Operating profit (after interest) divided by shareholders' accounting equity. |
| `ope_bel1` | Operating profits-to-lagged book equity | Same as `ope_be` but divided by last year's book equity. |
| `ni_be` | Return on equity (ROE) | Yearly net profit divided by shareholders' accounting equity. |
| `niq_be` | Quarterly return on equity | Latest quarter's net profit divided by book equity. |
| `niq_at` | Quarterly return on assets | Latest quarter's net profit divided by total assets. |
| `ebit_bev` | Return on net operating assets | Operating profit divided by the book value of the whole business (debt plus equity). |
| `ebit_sale` | Profit margin | Operating profit divided by sales: how many cents of profit per dollar of revenue. |
| `ocf_at` | Operating cash flow-to-assets | Cash generated by operations divided by total assets. |
| `at_turnover` | Asset turnover | Sales divided by assets: how hard the company's assets are working to generate revenue. |
| `sale_bev` | Sales-to-book enterprise value | Sales divided by the book value of the whole business (debt plus equity). |
| `opex_at` | Operating leverage | Operating expenses divided by assets; high means big fixed costs, so profits swing more with sales. |
| `pi_nix` | Taxable income-to-book income | Pre-tax income divided by after-tax net income, a hint of how much tax the company really pays. |

## 8. Growth & Earnings Surprises (15)

Whether sales and profits are improving, and whether the latest results beat what you'd have expected. Good surprises often lead to prices drifting up for months.

| Code | Name | What it means |
|---|---|---|
| `sale_gr1` | Sales growth (1 year) | Percentage growth in yearly revenue. |
| `sale_gr3` | Sales growth (3 years) | Percentage growth in yearly revenue over three years. |
| `saleq_gr1` | Quarterly sales growth | Latest quarter's revenue compared with the same quarter a year earlier. |
| `sale_emp_gr1` | Labour force efficiency | Growth in sales per employee over the past year. |
| `niq_at_chg1` | Change in quarterly return on assets | How much quarterly profit-to-assets improved versus the same quarter last year. |
| `niq_be_chg1` | Change in quarterly return on equity | How much quarterly return on equity improved versus the same quarter last year. |
| `ocf_at_chg1` | Change in operating cash flow-to-assets | How much operating cash flow (relative to assets) improved versus last year. |
| `niq_su` | Standardized earnings surprise | How far the latest quarter's profit beat or missed the same quarter last year, scaled by how bumpy earnings usually are. |
| `saleq_su` | Standardized revenue surprise | How far the latest quarter's sales beat or missed the same quarter last year, scaled by how bumpy sales usually are. |
| `ni_inc8q` | Consecutive earnings increases | Number of quarters in a row (up to 8) where profit rose versus the prior year. |
| `dsale_dinv` | Sales growth minus inventory growth | Positive when sales grow faster than unsold stock piling up in warehouses, a healthy sign. |
| `dsale_drec` | Sales growth minus receivables growth | Positive when sales grow faster than money customers still owe, suggesting real cash sales. |
| `dsale_dsga` | Sales growth minus overhead growth | Positive when sales grow faster than overhead costs like salaries and marketing. |
| `dgp_dsale` | Gross margin growth minus sales growth | Positive when the profit margin on products is improving faster than sales are growing. |
| `tax_gr1a` | Tax expense surprise | Change in taxes paid relative to assets; rising taxes often signal rising real profits. |

## 9. Investment & Asset Growth (14)

How fast the company is expanding. Firms that grow their asset base aggressively tend to have *lower* future returns.

| Code | Name | What it means |
|---|---|---|
| `at_gr1` | Asset growth | Percentage growth in total assets over the past year. |
| `be_gr1a` | Change in common equity | Change in shareholders' accounting equity over the past year, relative to assets. |
| `capx_gr1` | Capex growth (1 year) | Growth in spending on factories and equipment over one year. |
| `capx_gr2` | Capex growth (2 years) | Growth in spending on factories and equipment over two years. |
| `capx_gr3` | Capex growth (3 years) | Growth in spending on factories and equipment over three years. |
| `capex_abn` | Abnormal corporate investment | This year's equipment spending compared with the company's own recent average; high means an unusual spending spree. |
| `inv_gr1` | Inventory growth | Percentage growth in unsold goods held by the company. |
| `inv_gr1a` | Inventory change | Change in unsold goods over the year, relative to total assets. |
| `ppeinv_gr1a` | Change in PP&E and inventory | Change in property, plant, equipment and inventory, relative to total assets. |
| `noa_gr1a` | Change in net operating assets | Change in the assets used to run the business (net of operating liabilities), relative to total assets. |
| `lnoa_gr1a` | Change in long-term net operating assets | Change in long-lived operating assets like plants and equipment, relative to total assets. |
| `emp_gr1` | Hiring rate | Percentage growth in the number of employees. |
| `rd_sale` | R&D-to-sales | Research & development spending divided by revenue. |
| `rd5_at` | R&D capital-to-assets | Accumulated R&D spending over the past 5 years (older years count less) divided by total assets. |

## 10. Accruals & Balance-Sheet Changes (16)

Accruals are the gap between reported profit and cash. Profits that come mostly from accounting entries rather than cash tend to disappoint later. The `_gr1a` items break balance-sheet changes into their pieces.

| Code | Name | What it means |
|---|---|---|
| `oaccruals_at` | Operating accruals | Gap between operating profit and operating cash flow, divided by total assets. |
| `oaccruals_ni` | Percent operating accruals | Gap between operating profit and operating cash flow, as a share of net income. |
| `taccruals_at` | Total accruals | Gap between total profit and total cash flow (including investing and financing), divided by assets. |
| `taccruals_ni` | Percent total accruals | Gap between total profit and total cash flow, as a share of net income. |
| `noa_at` | Net operating assets | Operating assets minus operating liabilities, divided by total assets; bloated values hint at inflated past earnings. |
| `coa_gr1a` | Change in current operating assets | Change in short-term operating assets (inventory, money owed by customers) relative to total assets. |
| `col_gr1a` | Change in current operating liabilities | Change in short-term bills owed (to suppliers, employees) relative to total assets. |
| `cowc_gr1a` | Change in current operating working capital | Change in short-term operating assets minus short-term operating liabilities, relative to total assets. |
| `ncoa_gr1a` | Change in noncurrent operating assets | Change in long-term operating assets (equipment, intangibles) relative to total assets. |
| `ncol_gr1a` | Change in noncurrent operating liabilities | Change in long-term non-debt obligations (like pensions or deferred taxes) relative to total assets. |
| `nncoa_gr1a` | Change in net noncurrent operating assets | Change in long-term operating assets minus long-term operating liabilities, relative to total assets. |
| `lti_gr1a` | Change in long-term investments | Change in long-term financial investments the company holds, relative to total assets. |
| `sti_gr1a` | Change in short-term investments | Change in short-term financial investments the company holds, relative to total assets. |
| `fnl_gr1a` | Change in financial liabilities | Change in debt and other financial obligations, relative to total assets. |
| `nfna_gr1a` | Change in net financial assets | Change in financial investments minus financial debts, relative to total assets. |
| `debt_gr3` | Growth in book debt (3 years) | Percentage growth in the company's total debt over three years. |

## 11. Leverage & Financial Health (8)

How much the company relies on borrowing, how easily it could raise cash, and how likely it is to get into financial trouble.

| Code | Name | What it means |
|---|---|---|
| `at_be` | Book leverage | Total assets divided by shareholders' equity; high means the company is funded largely by debt. |
| `cash_at` | Cash-to-assets | Share of the company's assets held as cash. |
| `aliq_at` | Liquidity of book assets | How much of the company's assets could quickly be turned into cash, relative to total assets. |
| `aliq_mat` | Liquidity of market assets | How much of the company's assets could quickly be turned into cash, relative to its market-valued assets. |
| `tangibility` | Asset tangibility | How much of the company's assets are physical things that could be sold or borrowed against. |
| `kz_index` | Kaplan–Zingales index | A score for how financially constrained the company is; high means it struggles to raise money. |
| `z_score` | Altman Z-score | Classic bankruptcy-risk score; **higher means safer**. |
| `o_score` | Ohlson O-score | Another bankruptcy-risk score; **higher means more likely to go bust**. |

## 12. Earnings Stability (4)

How smooth, predictable and persistent the company's profits are.

| Code | Name | What it means |
|---|---|---|
| `earnings_variability` | Earnings variability | How much profit bounces around compared with how much cash flow does; high suggests less trustworthy earnings. |
| `ni_ar1` | Earnings persistence | How strongly this year's profit predicts next year's profit. |
| `ni_ivol` | Earnings volatility | How unpredictable profits are once you account for their usual year-to-year pattern. |
| `ocfq_saleq_std` | Cash flow volatility | How much quarterly cash flow (relative to sales) jumps around over time. |

## 13. Composite Quality Scores (7)

Ready-made scores that combine many of the signals above into a single "good company / bad company" rating.

| Code | Name | What it means |
|---|---|---|
| `f_score` | Piotroski F-score | A 0–9 checklist of financial health (profitable? cash flow positive? less debt? better margins?); higher is healthier. |
| `qmj` | Quality minus junk | An overall quality score combining profitability, growth and safety; higher is higher quality. |
| `qmj_prof` | Quality: profitability | The profitability part of the quality score. |
| `qmj_growth` | Quality: growth | The profit-growth part of the quality score. |
| `qmj_safety` | Quality: safety | The safety part of the quality score (low debt, low volatility, low bankruptcy risk). |
| `mispricing_mgmt` | Mispricing: management | A 0–1 ranking built from management decisions (share issuance, accruals, asset growth) that the market tends to misprice. |
| `mispricing_perf` | Mispricing: performance | A 0–1 ranking built from performance signals (profitability, momentum, distress) that the market tends to misprice. |

## 14. Market Risk (Beta) (6)

How tightly the stock is tied to the overall market. Surprisingly, high-beta stocks have historically not earned the extra return you'd expect.

| Code | Name | What it means |
|---|---|---|
| `beta_60m` | Market beta | How much the stock moved for each 1% move in the market, using the last 5 years of monthly returns. |
| `beta_dimson_21d` | Dimson beta | Last month's market beta, adjusted for stocks that react to market news a day late because they trade rarely. |
| `betabab_1260d` | Frazzini–Pedersen beta | A more stable beta built from 5 years of daily data by combining correlation and relative volatility. |
| `betadown_252d` | Downside beta | Beta measured only on days when the market fell: how badly the stock drops in bad times. |
| `corr_1260d` | Market correlation | How closely the stock's daily moves lined up with the market's over 5 years (−1 to 1). |
| `coskew_21d` | Coskewness | Whether the stock tends to crash especially hard when the market has extreme moves. |

## 15. Volatility & Lottery-Like Behaviour (12)

How wild the stock's price swings are. Very volatile, "lottery ticket" stocks with occasional huge up-days tend to be overpriced and earn *low* future returns.

| Code | Name | What it means |
|---|---|---|
| `rvol_21d` | Return volatility | How much the stock's daily price swung over the last month. |
| `ivol_capm_21d` | Idiosyncratic volatility (CAPM, 1 month) | Last month's daily price swings that were *not* explained by the overall market moving. |
| `ivol_capm_252d` | Idiosyncratic volatility (CAPM, 1 year) | Same as above but measured over the last year. |
| `ivol_ff3_21d` | Idiosyncratic volatility (FF3) | Last month's daily swings not explained by market, size or value trends. |
| `ivol_hxz4_21d` | Idiosyncratic volatility (q-factor) | Last month's daily swings not explained by market, size, investment or profitability trends. |
| `rskew_21d` | Return skewness | Over the last month, whether big daily moves were mostly jumps up (positive) or drops (negative). |
| `iskew_capm_21d` | Idiosyncratic skewness (CAPM) | Skewness of the company-specific part of daily returns after removing the market's effect. |
| `iskew_ff3_21d` | Idiosyncratic skewness (FF3) | Skewness of daily returns after removing market, size and value effects. |
| `iskew_hxz4_21d` | Idiosyncratic skewness (q-factor) | Skewness of daily returns after removing market, size, investment and profitability effects. |
| `rmax1_21d` | Maximum daily return | The single best daily return in the last month. |
| `rmax5_21d` | Average of top 5 daily returns | The average of the five best days in the last month. |
| `rmax5_rvol_21d` | Top-5 daily returns relative to volatility | The five best days' average divided by overall volatility; isolates "jackpot" behaviour from general jumpiness. |

## 16. Liquidity & Trading Activity (9)

How easy and cheap the stock is to trade. Hard-to-trade stocks can offer higher returns, but they are also costly (and sometimes impossible) to short in practice.

| Code | Name | What it means |
|---|---|---|
| `ami_126d` | Amihud illiquidity | How much the price moves per dollar traded; high means even small trades push the price around. |
| `bidaskhl_21d` | High–low bid-ask spread | An estimate of the gap between buying and selling prices, inferred from daily highs and lows; a direct trading cost. |
| `dolvol_126d` | Dollar trading volume | Average dollar value of shares traded per day over the last 6 months. |
| `dolvol_var_126d` | Variability of dollar volume | How much daily dollar trading volume fluctuates relative to its average. |
| `turnover_126d` | Share turnover | Average share of the company's stock that changes hands each day over 6 months. |
| `turnover_var_126d` | Variability of share turnover | How much daily turnover fluctuates relative to its average. |
| `zero_trades_21d` | Zero-trade days (1 month) | Number of days with no trading in the last month (ties broken by turnover); high means very illiquid. |
| `zero_trades_126d` | Zero-trade days (6 months) | Same as above over the last 6 months. |
| `zero_trades_252d` | Zero-trade days (12 months) | Same as above over the last 12 months. |
