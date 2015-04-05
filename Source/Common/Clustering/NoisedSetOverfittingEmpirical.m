function QEpsEmpirical = NoisedSetOverfittingEmpirical(L, ell, m1, m_r, r, d, eps)
    if (~exist('eps', 'var'))
        eps = 0.1;
    end

    A = false(d, L);

    if (nchoosek(m_r, r) < 1000)
        % create sample without replacement
        set = allsamples(m_r, r);
        set_len = size(set, 1);
        if (d > set_len)
            d = set_len;
        end
        
        set = set(randsample(set_len, d), :);

        A(:, 1 : m1) = true;
        for i = 1:d
            A(i, set(i, :) + m1) = true;
        end
    else
        % sample with replacement. When nchosek(m_r, r) >= 1000 this is
        % quite good aproximation of our algset.
        for i = 1:d
        	a0 = false(1, L);
            a0(1 : m1) = true;
            a0(m1 + randsample(m_r, r)) = true;
            A(i, :) = a0;
        end
    end    
    
    [~, ~, ~, QEpsEmpirical] = CalcOverfitting(AlgsetCreate(A), ell / L, 1000, eps, eps, eps);
end
