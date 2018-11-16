# RECOMMENDATION
I recommend using the epistasis software from the Harms lab, which has many more features.
https://epistasis.readthedocs.io/

# GlobalEpistasis

## install and load data
* Install julia (see julialang.org, on macos use homebrew). 
* Learn some julia (e.g. [wikibooks](https://en.wikibooks.org/wiki/Introducing_Julia))
* Update packages `Pkg.update()`
* add package `Pkg.clone("https://github.com/jotwin/GlobalEpistasis.jl")`, note this will add NLopt, CSV, and DataFrames packages
* I recommend [IJulia](https://github.com/JuliaLang/IJulia.jl) for an interactive notebook interface.

Load data into julia, [CSV doc](http://juliadata.github.io/CSV.jl/stable/). Test data is a subset of [Wu et al. eLife 2016](https://elifesciences.org/content/5/e16965/)
```
using CSV
using DataFrames
dataf = CSV.read("test/wu4sites_test.txt")
```
## prepare data

```
using GlobalEpistasis
d = prepdata(dataf, :mut, :sequence, "VDGV", :f, cname = :concentration, vname = :v)
```
arguments are

1. dataframe of your data
2. identifier of the column with sequence data
3. type of sequence data is either `:sequence` the sequence, or `:listofmuts` a list of substitutions where each substitution is a string with wild-type amino acid, position, and mutated amino acid, e.g. "V3A". Multiple substitutions per sequence can be concatenated together e.g. "V3A-G2B".
4. wild-type sequence, or if your data is in list of mutants format its the wild-type identifier, e.g. "WT"
5. name of the column with measured phenotype

optional keyword arguments are

* vname: name of the column of variance estimates
* cname: name of the column of conditions, which can be strings or numbers.
* delim: delimiter for multiple substitutions in listofmuts format. Default '-'
* condition_type: whether the condition variable is treated as a continuous or categorical predictor (`:categorical` or `:continuous`)

## non-epistatic model

```
@time mlin = nonepistatic_model(d)
```

output is a dictionary with keys:
* `:beta` is a data frame of additive effects
* `:r2` r-squared
* `:sigma2` HOC epistasis variance
* `:prediction` data frame of measured phenotype (y), inferred measured phenotype (yhat), and inferred additive trait (phi, rescaled), and (optional) conditions
* `:rmse` root mean squared error
* `:ll` log-likelihood

## global epistasis model

```
@time m = fit(mlin, nk = 4, tol = 1e-14)
```
* nk is number of knots in the I-spline (default 4)
* tol is the tolerance of the optimization (it can take a long time or fail if its too small)

output is a dictionary of keys similar to above

## writing output

write csv files

```
CSV.write("betas.txt", m[:beta])
CSV.write("prediction.txt", m[:prediction])

```

## modifying initial conditions
the fit function first argument is an initial model `m` with entries `m[:a]` and `m[:b]`. These can be modified to change the intial parameters. Also you can change the non-epistatic model e.g. `mlin[:a] = [0, 0, 0, 0, 0]`
