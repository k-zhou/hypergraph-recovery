# https://docs.python.org/3/library/math.html

# math.comb(n,k)    # binomial coefficient
# math.factorial(x)

L = 10 # max size of hyperedge, min 2
foldproduct = 1
for k in range(2, L + 1):
  N    = k # placeholder
  nu_k = 1 # placeholder
  E_k  = 1 # placeholder
  Z_k  = 1 # placeholder
  mu   = nu_k * comb(N, k)
  #print("k is", k, mu)
  P_H_k  = ( factorial(E_k)/( Z_k * pow(comb(N, k), E_k) *mu ) ) * pow(((N - k + 1)/k) + (1/mu), -E_k-1)
  foldproduct *= P_H_k

  #foldproduct *= k
  print("k is %3d P_H_k %1.6f and the rolling product is %2.10f " % (k, P_H_k, foldproduct))
