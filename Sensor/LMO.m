function w = LMO(grad,ksel,n)

[sorted_grad, sorted_indices] = sort(grad);
w = zeros(n,1);
for j = 1:ksel  % 只需要前5个小的
    w(sorted_indices(j)) = 1;
end

    


