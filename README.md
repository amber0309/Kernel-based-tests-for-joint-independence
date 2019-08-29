# Kernel‐based tests for joint independence

Python code of independence test algorithm proposed in

[Kernel‐based tests for joint independence](https://rss.onlinelibrary.wiley.com/doi/full/10.1111/rssb.12235?casa_token=1akkBcxMiBUAAAAA%3AG5ZNfHSt55CNjciCMT2R6uUTMx0RZ8ElretpE6jQgJbDkHombBp0OTG_oIkeqAOhlZ-u6Q5GYjsyG3tGcg)  
Niklas Pfister, Peter Bühlmann, Bernhard Schölkopf, Jonas Peters  
*Journal of the Royal Statistical Society: Series B (Statistical Methodology)* 80.1 (2018): 5-31.

## Prerequisites

- numpy (=1.13.3)
- scipy (=1.1.0)
- statsmodels (=0.10.1)

We test the code using python 3.6.8 on Windows 10. Any later version should still work perfectly.

## Running the test

After installing all required packages, you can run *demo.py* to see whether `dhsic` could work normally.

The test code does the following:

1. generates 100 instances (a (100, 3) *numpy array*) from 3 jointly independent variables;
2. apples joint independence test on the data
3. changes variable 3 to be dependent on variable 1 and 2
4. applies joint independence test on the updated data.

## Apply `dhsic` on your data

### Usage

```python
from dhsic import dhsic_test, dhsic

res = dhsic_test(X)
res = dhsic(X)
```

### Description

Function `dhsic_test()`

| Argument  | Description  |
|---|---|
|X | matrix of all instances, (n_samples, n_vars) numpy array |
|alpha (optional) | level of the test |
|method (optional) | method of the test |

| Output  | Description  |
|---|---|
| res | distionary containing test statistic, critical value, p-value and the method name |

Function `dhsic()`

| Argument  | Description  |
|---|---|
|X | matrix of all instances, (n_samples, n_vars) numpy array |

| Output  | Description  |
|---|---|
| dHSIC | dHSIC value |

## Author

- **Shoubo Hu** - shoubo [dot] sub [at] gmail [dot] com

See also the list of [contributors](https://github.com/amber0309/Kernel-based-tests-for-joint-independence/graphs/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
