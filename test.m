%%Linear equations system construction
n = 3;
A = rand(n,n);
A = A'*A;  %To have a symetric positive definite matrix
sol = rand(n,1);
b = A * sol;
tol = 1.d-16;
itermax = 100;
x0 = zeros(n,1);

[x,~] = bi_conjugate_gradient(A,b,x0,tol,itermax);
residual = A*x-b;
fprintf('Bi-conjugate gradient residual: %e\n',norm(residual));

[x,~,~] = conjugate_gradient(A,b,x0,tol,itermax);
residual = A*x-b;
fprintf('Conjugate gradient residual: %e\n',norm(residual));

