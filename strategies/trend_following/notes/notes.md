# Notes
write this first to clear my thoughts

The goal is to take as many independent uncorrelated bets as possible across the whole tradeable universe.
This is where universe selection comes in

IR = TC * IC * sqrt(BR)

Neff?

(Fast ma log price - slow ma log price)/vol

trend horizon: 1m - 12m ; avg 3-6m

trend signal passes through function to map signal to target risk allocation for that market, allocation should cap out at some point ; maps trend strength to target exposure

sigmoid func
x * exp(-x^2) ; overextended markets

add bias? to this func for markets to go up on avg

trend strength -> target exposure -> size positions

positions inversely proportional to a market-specific volatility forecast
steady_state long term vol and short term realised vol (30-60days)

sector specific weights
simple equal weighting, complex hierarchical,pca
bucket universe into broad asset class groups, try and have roughly equal vol from each asset class in the long run

portfolio risk targeting 0.3x - 3x risk target
run more risk when not many markets are trending, lesser risk when more markets are trending
size up positions when there are few trends, size down when there are more trends
adjust based on what u want to achieve

dont need to trade on every change in signal or vol forecast, use buffering
alpha is uncertain but trading cost (commisions/slippage) is known. keep trading to minimum to monetize alpha
(read tmr in cephalopod post)

keep signals simple to reduce overfit
dont concentrate positions too much
trade as wide a universe as possible
maximise diversification
minimise trading

VIX ?
generalise trend to directional
bring in models non linear in nature to produce signals that dont look like trend in the typical sense
sizing: combine signal strength and BARRA system

cephalopod
advanced pm
linkedin quant pm post