# GlobalEpistasis

## install and load data
* Install julia (see julialang.org, on macos use homebrew). 
* Learn some julia (e.g. [wikibooks](https://en.wikibooks.org/wiki/Introducing_Julia))
* Update packages `Pkg.update()`
* add packages `Pkg.add("NLopt")`, `Pkg.add("DataFrames")`

Load data into julia
```
Pkg.add("CSV") # only first time
using CSV
using DataFrames
mydata = CSV.read("yourfile.txt", delim="\t")
```
## prepare data

```
include("globalepistasis.jl")
d = prepdata(muts, phenotype, v=v, c=c, delim='-', wt = "WT", condition_type = :categorical)
```
* muts is a vector of strings that describe the mutations with wild-type amino acid, position, and mutated amino acid, like "K25Q". Mutations can be put together on one line as in "K25Q-G35E". The optional delim parameter specifies the delimeter (default `'-'`). A wild-type sequence can be included by specifying the wild-type string with optional parameter wt (default `"WT"`).
* phenotype is a vector of measured phenotypes
* v is the optional variance estimates. Missing values should be set to 0.0
* c is the optional experimental condition vector, which can be strings or numbers. condition_type defines whether the condition variable is treated as a continuous or categorical predictor (`:categorical` or `:continuous`)

## non-epistatic model

```
@time mlin = nonepistatic_model(d)
```

output is a dictionary with keys:
* `:b` coefficients
* `:r2` r-squared
* `:sigma2` HOC epistasis variance
* `:yhat` inferred measured phenotype
* `:rmse` root mean squared error
* `:ll` log-likelihood

## global epistasis model

```
@time m = spmfit(mlin, nk = 4, tol = 1e-14)
```
* nk is number of knots in the I-spline (default 4)
* tol is the tolerance of the optimization (it can take a long time or fail if its too small)

output is a dictionary of keys similar to above, but also
* `:bg` are the scaled coefficients of the additive trait
* `:phi` and `:phiG` are the raw and scaled values of the additive trait

