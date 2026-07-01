import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

errors = np.loadtxt("model_results.txt")
lin_test_error, rbf_test_error, nn_test_error = errors

n_test = 2000
z = 1.96

def wald_ci(error, n):
    half = z * np.sqrt(error * (1 - error) / n)
    return error - half, error + half

results = [('Linear SVM', lin_test_error), ('Gaussian SVM (RBF)', rbf_test_error), ('Neural Network', nn_test_error)]

comp = []
for name, err in results:
    lo, hi = wald_ci(err, n_test)
    comp.append({'method': name, 'test_err': err, 'ci_low': lo, 'ci_high': hi,
                 'ci_halfwidth': z*np.sqrt(err*(1-err)/n_test)})
comp = pd.DataFrame(comp); print(comp)

plt.figure(figsize=(7, 4))
ys = np.arange(len(comp))
plt.errorbar(comp['test_err'], ys, xerr=comp['ci_halfwidth'], fmt='o', capsize=5)
plt.yticks(ys, comp['method'])
plt.xlabel('Test error (95% Wald CI)')
plt.title('Comparison of three tuned models')
plt.grid(True, axis='x'); plt.tight_layout()
plt.savefig('part4 comparison.png', dpi=800, bbox_inches='tight'); plt.show()

print('Pairwise CI overlap (True = NOT clearly distinguishable):')
for i in range(len(comp)):
    for j in range(i+1, len(comp)):
        a, b = comp.iloc[i], comp.iloc[j]
        overlap = not (a['ci_high'] < b['ci_low'] or b['ci_high'] < a['ci_low'])
        print(f"  {a['method']:20s} vs {b['method']:20s}: overlap = {overlap}")