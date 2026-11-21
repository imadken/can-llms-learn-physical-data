from effect_pretraining.baseline import wilson_interval
for n in range(1,21):
 for k in range(n+1):
  lo,hi=wilson_interval(k,n)
  if not lo<=k/n<=hi: print('interval-roundoff',k,n,lo,hi)
print('interval arithmetic checked for possible 1–20 valid episodes')
