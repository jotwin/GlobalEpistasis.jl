module GlobalEpistasis

export prepdata, nonepistatic_model, spmfit

using NLopt
using DataFrames

function prepdata(muts, y; delim = '-', wt = "WT", c = nothing, v = nothing, condition_type = :categorical)
    n = length(muts)
    I = collect(1:n)
    J = ones(Int64, n)
    j = 2
    aa = Char[]
    pos = Int64[]
    code = Dict{String, Int64}()
    
    for i = 1:n
        if muts[i] != wt
            for k in split(muts[i], delim)
                if !haskey(code, k)
                    code[k] = j
                    push!(aa, k[end])
                    push!(pos, parse(k[2:end-1]))
                    j += 1
                end
                push!(J, code[k])
                push!(I, i)
            end
        end
    end
    x = sparse(I, J, 1.0)
    g = trues(size(x,2))
    g[1] = false
    ham = ceil.(Int64, vec(sum(x,2))-1)
    dout = Dict(:x => x, :y => y, :code => code, :g => g,
                :ham=> ham, :pos => pos, :aa => aa)
    if v != nothing
        dout[:v] = v#collect(Missings.replace(v, 0.0))
    end
	if c != nothing
		if condition_type == :categorical
			lev1 = levels(c)[1]
			for lev in levels(c)[2:end]
				x = hcat(x, sparse(c .== lev))
				push!(g, false)
			end
		elseif condition_type == :continuous
			c = convert(Vector{Float64}, c)
			x = hcat(x, c)
			push!(g, false)
		end
		dout[:c] = c
		dout[:x] = x
		dout[:g] = g
	end
    return dout
end
function nonepistatic_model(data; kwargs...)
    x = data[:x]
    y = data[:y]
    b = x \ y
    yhat = x * b
    mse = mean((y-yhat).^2)
    r2 = cor(y, yhat)^2
    ll = -length(y)/2 - 1/2*log(mse)
    m = Dict(:phi => yhat, :yhat => yhat, :b => b, :sigma2=>mse, :sigma2p => 1e-6,
        :method => :leftdiv, :data => data, :r2 => r2, :ll => ll)
    if haskey(data, :c)
	    g = data[:g]
	    m[:phiG] = view(x, :, g) * view(b, g)
    end
    if haskey(data, :v)
        return spmopt(m, false, true; kwargs...)
    else
        return m
    end
end

function monosplinebasis1(x, knots, k)
  @assert all(diff(knots) .> 0) "knots must be increasing"
  @assert length(knots) > 2 "must have at least 3 knots"
  nk = length(knots)
  na = nk+k-1
  I = zeros(na)
  I[1] = 1.0
  M = zeros(na)
  t = [[knots[1] for j in 1:k-1]; knots; [knots[end] for j in 1:k-1]]
  slope1, slope2 = monosplinebasisslopes(t, na-1, k)

  Mall = zeros(length(x), na)
  Iall = zeros(length(x), na)
  Iall[:,1] = 1.0
  # monosplinebasis!(Mall, Iall, slope1, slope2, x, t, k)
  for (i,xi) in enumerate(x)
    monosplinebasis1!(M, I, slope1, slope2, xi, t, k)
    Mall[i,:] = M
    Iall[i,:] = I
  end
  return (Mall, Iall, t, slope1, slope2)
end

function monosplinebasisslopes(t, na, kk)
  slope = zeros(2, na)
  for i in 1:na
    if t[i] < t[i+1]
      slope[1, i] = 1.0/(t[i+1]-t[i])
      slope[2, i] = 1.0/(t[i+1]-t[i])
    end
  end
  for k in 2:kk
    for i in 1:na
      if t[i] < t[i+k]
        slope[1, i] = (t[1]-t[i])*slope[1, i]*k/(k-1)/(t[i+k]-t[i])
        slope[2, i] = (t[end]-t[i])*slope[2, i]*k/(k-1)/(t[i+k]-t[i])
        if i+1 < na
          slope[1, i] += (t[i+k]-t[1])*slope[1, i+1]*k/(k-1)/(t[i+k]-t[i])
          slope[2, i] += (t[i+k]-t[end])*slope[2, i+1]*k/(k-1)/(t[i+k]-t[i])
        end
      end
    end
  end
  s1 = slope[1,1]
  s2 = slope[2,end]
  return (s1, s2)
end

function monosplinebasis1!(M::Vector{Float64}, I::Vector{Float64}, slope1::Float64, slope2::Float64, x::Float64, t::Vector{Float64}, kk::Int64)
  # M spline
  na = length(M) - 1
  M[2:na+1] = 0.0
  # extend basis to extrapolate linearly beyond knots
  if x <= t[1]
    # M[3:end] = 0.0
    M[2] = slope1
    I[3:end] = 0.0
    I[2] = slope1 * (x-t[1])
  elseif x >= t[end]
    # M[2:na] = 0.0
    M[na+1] = slope2
    I[2:na] = 1.0
    I[na+1] = 1.0 + slope2 * (x-t[end])
  else
    for i in 1:na
      ti1 = 1.0/(t[i+1]-t[i])
      if t[i] < x <= t[i+1]# || x[j] == t[1]
        M[i+1] = ti1
      # else
      #   M[i+1] = 0.0
      end
    end
    for k in 2:kk
      for i in 1:na
        tikk = k/(k-1)/(t[i+k]-t[i])
        m = 0.0
        if t[i] < x <= t[i+k]
          if x <= t[i+k-1]
            m += (x-t[i])*M[i+1]
          end
          if t[i+1] < x
            m += (t[i+k]-x)*M[i+2]
          end
          m *= tikk
        end
        M[i+1] = m
      end
    end
    # I spline
    nt = length(t)
    for i in 1:na
      j = 1
      for jj in 2:nt
        if x <= t[jj]
          j = jj-1
          break
        end
      end
      Itemp = 0.0
      if i > j
        Itemp = 0.0
      elseif i < j - kk + 1
        Itemp = 1.0
      else
        for m in i:j
          if m+1 <= na
            Itemp += 1/kk*((x-t[m])*M[m+1] + (t[m+kk+1]-x)*M[m+2])
          else
            Itemp += 1/kk*((x-t[m])*M[m+1])
          end
        end
      end
      I[i+1] = Itemp
    end
  end
end

function monosplinebasis!(M, I, slope1, slope2, xv, t, kk)
  # M spline
  na = size(M,2)-1
  M[:,2:na+1] = 0.0
  for (p, xp) in enumerate(xv)
#     Threads.@threads for p = 1:length(xv)
#     xp = xv[p]
    # extend basis to extrapolate linearly beyond knots
    if xp <= t[1]
      # M[3:end] = 0.0
      M[p,2] = slope1
      I[p,3:end] = 0.0
      I[p,2] = slope1 * (xp-t[1])
    elseif xp >= t[end]
      # M[2:na] = 0.0
      M[p,na+1] = slope2
      I[p,2:na] = 1.0
      I[p,na+1] = 1.0 + slope2 * (xp-t[end])
    else
      for i in 1:na
        ti1 = 1.0/(t[i+1]-t[i])
        if t[i] < xp <= t[i+1]# || x[j] == t[1]
          M[p,i+1] = ti1
        # else
        #   M[i+1] = 0.0
        end
      end
      for k in 2:kk
        for i in 1:na
          tikk = k/(k-1)/(t[i+k]-t[i])
          m = 0.0
          if t[i] < xp <= t[i+k]
            if xp <= t[i+k-1]
              m += (xp-t[i])*M[p,i+1]
            end
            if t[i+1] < xp
              m += (t[i+k]-xp)*M[p,i+2]
            end
            m *= tikk
          end
          M[p,i+1] = m
        end
      end
      # I spline
      nt = length(t)
      for i in 1:na
        j = 1
        for jj in 2:nt
          if xp <= t[jj]
            j = jj-1
            break
          end
        end
        Itemp = 0.0
        if i > j
          Itemp = 0.0
        elseif i < j - kk + 1
          Itemp = 1.0
        else
          for m in i:j
            if m+1 <= na
              Itemp += 1/kk*((xp-t[m])*M[p,m+1] + (t[m+kk+1]-xp)*M[p,m+2])
            else
              Itemp += 1/kk*((xp-t[m])*M[p,m+1])
            end
          end
        end
        I[p,i+1] = Itemp
      end
    end
  end
end

function spmopt(mi, estimate_alpha = haskey(mi, :a), estimate_beta = haskey(mi, :b); 
        nk = 4, knots = :linear, 
        a_upper_bound = get(mi, :a_upper_bound, [Inf, Inf]), 
		a_lower_bound = get(mi, :a_lower_bound, [0.0, 0.0]),
        maxit = 1000000, alg = :LD_LBFGS, tol=1e-14)

    
    data = mi[:data]
    x = data[:x]
    y = data[:y]
    
    n, nb = size(x)
    pi = zeros(0)
    lbounds = zeros(0)
    ubounds = zeros(0)

    estimate_sigma2 = haskey(data, :v)
    estimate_sigma2p = false
    if estimate_sigma2
        v = data[:v]
        estimate_sigma2p = any(v .== 0.0)
        push!(pi, log(mi[:sigma2]))
        push!(lbounds, -Inf)
        push!(ubounds, Inf)
        if estimate_sigma2p
            push!(pi, log(mi[:sigma2p]))
            push!(lbounds, -Inf)
            push!(ubounds, Inf)
        end
        iv0 = find(v .== 0.0)
        iv = find(v .!= 0.0)
    else
        iv0 = nothing
        iv = nothing
	    v = nothing
    end
    
    phi = copy(mi[:phi])
    b = copy(mi[:b])
    if estimate_alpha && !estimate_beta
        minphi, maxphi = extrema(phi)
        # rescale phi to have range 0..1
        b = b/(maxphi-minphi)
        b[1] = b[1] - minphi/(maxphi-minphi)
        phi = (phi-minphi)/(maxphi-minphi)
    end
    if estimate_alpha
        if haskey(mi, :knots)
            knots = mi[:knots]
        elseif knots == :linear
            knots = linspace(0, 1, nk)
        elseif knots == :quantile
            knots = unique(quantile(phi,  linspace(0, 1, nk)))
            if length(knots) < nk
                warn("knots removed  from quantile redundancy")
            end
        end
        M, I, t, slope1, slope2 = monosplinebasis1(phi, knots, 3)
        na = size(I,2)
        a = zeros(na)
        arange = length(pi) + (1:na)
        if haskey(mi, :a)
            #push!(pi, mi[:a][1])
            #append!(pi, log.(mi[:a][2:end]))
            a0 = copy(mi[:a])
            if a_upper_bound[1] < a0[2]
                a0[2] = a_upper_bound[1]
            end
            if a_upper_bound[2] < a0[end]
                a0[end] = a_upper_bound[2]
            end
            append!(pi, a0)
        else
            #push!(pi, 0.0)
            #append!(pi, log.(0.001*ones(na-1)))
            append!(pi, zeros(na))
        end
        append!(lbounds, [-Inf; a_lower_bound[1]; zeros(na-3); a_lower_bound[2]])
        append!(ubounds, [Inf; a_upper_bound[1]; Inf*ones(na-3); a_upper_bound[2]])
    else
        arange = nothing
        knots = nothing
        M = nothing
        I = nothing
        t = nothing
        slope1 = nothing
        slope2 = nothing
        a = nothing
    end
    if estimate_beta
        brange = length(pi) + (1:nb)
        append!(pi, b)
        append!(lbounds, -Inf*ones(nb))
        append!(ubounds, Inf*ones(nb))
    else
        brange = nothing
    end
    opt = Opt(alg, length(pi))
    lower_bounds!(opt, lbounds)
    upper_bounds!(opt, ubounds)
    maxeval!(opt, maxit)
    ftol_rel!(opt, tol)
    
    yhatp = ones(y)
    llmem = similar(y)
    gs2 = similar(y)
    gs2p = similar(y)
    rsy = similar(y)
    rsvi = similar(y)

    cb = (p, g) ->  spm_objective(p, g, x, y, v, 
            estimate_sigma2, estimate_sigma2p, estimate_alpha, estimate_beta,
            arange, brange, M, I, t, slope1, slope2, 
            phi, yhatp, llmem, gs2, gs2p, rsy, rsvi, iv0, iv)
    max_objective!(opt, cb)
    (ll, p, ret) = optimize!(opt, pi)
    #if estimate_alpha && estimate_beta
    #    a = p[arange]
    #    if a[2] < 0.01 || a[end] < 0.01
    #        display("reoptimizing with lower a bound 0.01")
    #        pi = p
    #        ai = [arange[2], arange[end]]
    #        lbounds[ai] = 0.01
    #        pi[ai] = 0.01
    #        lower_bounds!(opt, lbounds)
    #        (ll, p, ret) = optimize!(opt, pi)
    #        display("reoptimizing with lower a bound 0.0")            
    #        pi = p
    #        lbounds[ai] = 0.0
    #        lower_bounds!(opt, lbounds)
    #        (ll, p, ret) = optimize!(opt, pi)
    #    end
    #end
    g = data[:g]
	if estimate_beta && estimate_alpha && all(data[:ham] .<= 1) # single mutants multi conditions
		betabad = p[brange]
		phiE = x[:, .!g] * betabad[.!g]
	    constrain_betas!(p[arange], betabad, g, phiE, 0.1)
		p[brange] = betabad
		#display("constraining betas, reoptimizing")
    	#(ll, p, ret) = optimize!(opt, p)
		#constrain_betas!(p[arange], betabad, g, phiE, 0.1)
		#p[brange] = betabad
	end

    
    m = Dict(:data => data, :nlopt_return => ret, :ll => ll*n)
    if estimate_beta
        m[:b] = p[brange]
    else
        m[:b] = b
    end
    if haskey(data, :c)
        #phiE = view(x, :, .!g) * m[:b][.!g]
	    m[:be] = m[:b][.!g]
	    m[:phiE] = x[:, .!g] * m[:be]
    end
    m[:bg] = m[:b][g]/mean(abs.(m[:b][g]))
    m[:phi] = x*m[:b]
    m[:phiG] = x[:, g] * m[:bg]

    
    if estimate_alpha
        m[:a] = p[arange]
        m[:yhat] = monosplinebasis1(m[:phi], knots, 3)[2]*m[:a]
        m[:knots] = knots
        m[:a_upper_bound] = a_upper_bound
    else
        m[:yhat] = m[:phi]
    end
    m[:r2] = cor(y, m[:yhat])^2
	m[:rmse] = sqrt(mean((m[:yhat]-data[:y]).^2))
    if estimate_sigma2
        m[:sigma2] = exp(p[1])
        if estimate_sigma2p
            m[:sigma2p] = exp(p[2])
        end
    else
        m[:sigma2] = mean((y.-m[:yhat]).^2)
        m[:ll] = -n/2*log(m[:sigma2])
    end
    
    return m
end

function constrain_betas!(a, b, g, phiE, delta = 0.0)
  if a[end] == 0.0 # curve is flat on right
    minbc = minimum(phiE)
    for i in find(g)
      if minbc + b[i] > 1
        b[i] = 1 - minbc - delta
      end
    end
  end
  if a[2] == 0.0 # curve is flat on left
    maxbc = maximum(phiE)
    for i in find(g)
      if maxbc + b[i] < 0
        b[i] = -maxbc + delta
      end
    end
  end
end




function spm_objective(p, g, x, y, v, 
            estimate_sigma2, estimate_sigma2p, estimate_alpha, estimate_beta,
            arange, brange, M, I, t, slope1, slope2, 
            yhat, yhatp, ll::Vector{Float64}, gs2, gs2p, rsy, rsvi::Vector{Float64}, iv0, iv)  
    if estimate_beta
        A_mul_B!(yhat, x, view(p, brange))
    end
    if estimate_alpha
        #a .= exp.(view(p, arange))
        #a[1] = p[arange[1]]
        #for i = 2:length(a)
        #    ai = p[arange[i]]
        #    if ai > 20
        #        ai = 20
        #    end
        #    a[i] = exp(ai)
        #end
        a = view(p, arange)
        if estimate_beta
            monosplinebasis!(M, I, slope1, slope2, yhat, t, 3)
        end
        A_mul_B!(yhat, I, a)
        A_mul_B!(yhatp, M, a)
    end
    
    n, nb = size(x)
#     Threads.@threads for j = 1:n
    if estimate_sigma2
        s = exp(p[1])
        for j in iv
	        r = y[j]-yhat[j]
	        svi = 1.0/(s+v[j])
	        rsvi[j] = r*svi
	        gs2p[j] = 0.0
	        gs2[j] = (rsvi[j]*r-1)*svi*s/2
	        ll[j] = -rsvi[j]*r-log(s+v[j])
        end
        if estimate_sigma2p
            s2 = exp(p[2])
            svi2 = 1/(s+s2)
            lsvi2 = log(s+s2)
            for j in iv0
                r = y[j]-yhat[j]
                rsvi[j] = r*svi2
                gsvi = (rsvi[j]*r-1)*svi2/2
                gs2p[j] = gsvi*s2
                gs2[j] = gsvi*s
                ll[j] = -rsvi[j]*r-lsvi2
            end
            g[2] = sum_kbn(gs2p)
        end
        g[1] = sum_kbn(gs2)
    else
        for j = 1:n
            r = y[j]-yhat[j]
            rsvi[j] = r
            ll[j] = -r^2
        end
    end
    sll = sum_kbn(ll)/2/n
   # display(any(isnan.(yhat)))
#     if isnan(sll)
#     #    display("$s $s2")
#         display(p[brange])
#         display(p[arange])
#         display(find(isnan.(yhat)))
#         display(find(isnan.(p[brange])))
#         #display(find(isnan.(p[arange])))
#         error("nan ll")
#     end
    if estimate_beta
        rsy .= rsvi.*yhatp
        At_mul_B!(view(g, brange), x, rsy)
        #display("$(p[brange[177]]) $(g[brange[177]])")
    end
    if estimate_alpha
        At_mul_B!(view(g, arange), I, rsvi)
        #for i = 2:length(a)
        #    g[arange[i]] = g[arange[i]]*a[i]
        #end
    end
   #display(exp.(p[1:2]))
    g .= g/n
	#display(g[end])
    return sll

end

function spmfit(m0::Dict; kwargs...)
	if !haskey(m0, :a)
	    m0 = spmopt(m0, true, false; kwargs...)
	end
    spmopt(m0, true, true; kwargs...)
end

end # module
